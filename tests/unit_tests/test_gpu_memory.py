# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage-tagged GPU/PID telemetry with real files and fake hardware APIs."""

import json
import os
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from omegaconf import OmegaConf

from rlinf.config import memory_logging_options, validate_gpu_memory_logging_cfg
from rlinf.scheduler import AcceleratorType, Worker, WorkerAddress
from rlinf.scheduler.hardware.accelerators.nvidia_memory import NvidiaMemorySampler
from rlinf.utils.gpu_memory import (
    DeviceMemoryMonitor,
    TorchMemoryRecorder,
    host_memory_directory,
    logical_payload_size,
    start_host_memory_helper,
)
from rlinf.utils.host_memory import (
    HostMemorySampler,
    JsonlWriter,
    ProcessMemorySampler,
    atomic_json,
    identity_alive,
    proc_identity,
    read_json,
)


def _options(tmp_path):
    return {
        "output_dir": str(tmp_path),
        "run_id": "test_run",
        "torch": True,
        "nvml": True,
        "interval_s": 0.01,
    }


def _metadata():
    return {
        "node_rank": 0,
        "component": "rollout",
        "rank": 2,
        "child_rank": None,
        "role": "main_tokenizer_manager",
        "worker_address": "rollout:2",
        "pid": os.getpid(),
        "create_time": psutil.Process().create_time(),
    }


def _nvml(monkeypatch, entries=None, sampled=None):
    import pynvml as nvml

    monkeypatch.setattr(nvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(nvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(nvml, "nvmlDeviceGetCount", lambda: 1)
    monkeypatch.setattr(nvml, "nvmlDeviceGetHandleByIndex", lambda index: index)
    monkeypatch.setattr(nvml, "nvmlDeviceGetUUID", lambda handle: b"GPU-test")
    monkeypatch.setattr(
        nvml,
        "nvmlDeviceGetPciInfo",
        lambda handle: SimpleNamespace(busId=b"0000:01:00.0"),
    )

    def memory(handle):
        if sampled is not None:
            sampled.set()
        return SimpleNamespace(total=1000, used=700, free=300)

    monkeypatch.setattr(nvml, "nvmlDeviceGetMemoryInfo", memory)
    monkeypatch.setattr(
        nvml,
        "nvmlDeviceGetComputeRunningProcesses",
        lambda handle: entries or [SimpleNamespace(pid=os.getpid(), usedGpuMemory=400)],
    )
    monkeypatch.setattr(
        nvml,
        "nvmlDeviceGetGraphicsRunningProcesses",
        lambda handle: [SimpleNamespace(pid=os.getpid(), usedGpuMemory=400)],
    )
    return nvml


def test_nvml_deduplicates_compute_graphics_and_preserves_unavailable(monkeypatch):
    _nvml(
        monkeypatch,
        [
            SimpleNamespace(pid=os.getpid(), usedGpuMemory=400),
            SimpleNamespace(pid=99999999, usedGpuMemory=(1 << 64) - 1),
        ],
    )
    sampler = NvidiaMemorySampler()
    try:
        device = sampler.sample()[0]
    finally:
        sampler.close()
    processes = {process["pid"]: process for process in device["processes"]}
    assert device["gpu_uuid"] == "GPU-test"
    assert processes[os.getpid()]["used_bytes"] == 400
    assert processes[os.getpid()]["kinds"] == ["compute", "graphics"]
    assert processes[99999999]["used_bytes"] is None


def test_nvml_query_failure_keeps_device_and_other_process_list(monkeypatch):
    nvml = _nvml(monkeypatch)

    def unavailable(handle):
        raise nvml.NVMLError_NotSupported()

    monkeypatch.setattr(nvml, "nvmlDeviceGetComputeRunningProcesses", unavailable)
    sampler = NvidiaMemorySampler()
    try:
        device = sampler.sample()[0]
    finally:
        sampler.close()
    assert device["used_bytes"] == 700
    assert device["processes"][0]["kinds"] == ["graphics"]
    assert len(device["process_query_errors"]) == 1


def test_node_monitor_labels_descendants_and_stops_periodic_writer(
    tmp_path, monkeypatch
):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    sampled = threading.Event()
    _nvml(
        monkeypatch,
        [SimpleNamespace(pid=child.pid, usedGpuMemory=100)],
        sampled,
    )
    options = _options(tmp_path)
    directory = tmp_path / "test_run/processes/node_0"
    directory.mkdir(parents=True)
    (directory / f"{os.getpid()}.json").write_text(json.dumps(_metadata()))
    monitor = DeviceMemoryMonitor(options, 0, AcceleratorType.NV_GPU)
    try:
        monitor.start()
        monitor.set_stage(16, "actor_training")
        sampled.clear()
        assert sampled.wait(timeout=2)
        monitor.stop()
        monitor.stop()
        path = next((tmp_path / "test_run/nvml").glob("*.jsonl"))
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records[-1]["event"] == "monitor_stop"
        assert any(
            record["event"] == "periodic" and record["stage"] == "actor_training"
            for record in records
        )
        boundary = next(
            record for record in records if record["event"] == "stage_boundary"
        )
        processes = {p["pid"]: p for p in boundary["devices"][0]["processes"]}
        assert processes[child.pid]["component"] == "rollout"
        assert processes[child.pid]["rank"] == 2
        assert processes[child.pid]["attribution"] == "ancestor"
        assert processes[child.pid]["role"] == "child_process"
        size = path.stat().st_size
        assert monitor._stop.wait(timeout=0.02)
        assert path.stat().st_size == size
    finally:
        monitor.stop()
        child.terminate()
        child.wait(timeout=5)


def test_monitor_failure_is_persisted_without_interrupting_training(
    tmp_path, monkeypatch
):
    nvml = _nvml(monkeypatch)

    def unavailable():
        raise nvml.NVMLError_DriverNotLoaded()

    monkeypatch.setattr(nvml, "nvmlInit", unavailable)
    monitor = DeviceMemoryMonitor(_options(tmp_path), 0, AcceleratorType.NV_GPU)
    monitor.start()
    monitor.stop()
    record = json.loads(next((tmp_path / "test_run/nvml").glob("*.jsonl")).read_text())
    assert record["record_type"] == "monitor_error"


def test_torch_uninitialized_does_not_query_or_initialize_devices(tmp_path):
    class UninitializedPlatform:
        def is_initialized(self):
            return False

        def __getattr__(self, name):
            raise AssertionError(f"Unexpected device call: {name}")

    recorder = TorchMemoryRecorder(
        _options(tmp_path), _metadata(), UninitializedPlatform()
    )
    recorder.set_stage(1, "rollout", reset_peak=True)
    with recorder.phase("offload"):
        pass
    records = [
        json.loads(line)
        for line in next((tmp_path / "test_run/torch/node_0").glob("*.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert all(
        not record["device_initialized"] and record["devices"] == []
        for record in records
    )


def test_torch_phase_records_allocator_peak_and_failure(tmp_path):
    class Platform:
        allocated = 100
        reserved = 300
        peak_allocated = 900
        peak_reserved = 1000

        def is_initialized(self):
            return True

        def device_count(self):
            return 1

        def memory_allocated(self, device):
            return self.allocated

        def memory_reserved(self, device):
            return self.reserved

        def max_memory_allocated(self, device):
            return self.peak_allocated

        def max_memory_reserved(self, device):
            return self.peak_reserved

        def reset_peak_memory_stats(self, device):
            self.peak_allocated, self.peak_reserved = self.allocated, self.reserved

    platform = Platform()
    recorder = TorchMemoryRecorder(_options(tmp_path), _metadata(), platform)
    recorder.set_stage(16, "actor_training", reset_peak=True)
    with pytest.raises(RuntimeError, match="simulated OOM"):
        with recorder.phase("actor/forward_backward"):
            platform.peak_allocated = 800
            raise RuntimeError("simulated OOM")
    records = [
        json.loads(line)
        for line in next((tmp_path / "test_run/torch/node_0").glob("*.jsonl"))
        .read_text()
        .splitlines()
    ]
    failure = next(record for record in records if record["event"].endswith("/error"))
    assert failure["step"] == 16
    assert failure["devices"][0]["allocated_bytes"] == 100
    assert failure["devices"][0]["reserved_minus_allocated_bytes"] == 200
    assert failure["devices"][0]["max_allocated_bytes"] == 800
    assert failure["peak_scope"] == "actor/forward_backward"


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_memory_config_rejects_invalid_interval(interval):
    cfg = OmegaConf.create(
        {"runner": {"gpu_memory_logging": {"enabled": True, "interval_s": interval}}}
    )
    with pytest.raises(ValueError, match="finite and positive"):
        validate_gpu_memory_logging_cfg(cfg)


def test_memory_config_defaults_and_preserves_explicit_values(tmp_path):
    cfg = OmegaConf.create({"runner": {"logger": {"log_path": str(tmp_path)}}})
    validate_gpu_memory_logging_cfg(cfg)
    assert not cfg.runner.gpu_memory_logging.enabled
    cfg.runner.gpu_memory_logging.enabled = True
    cfg.runner.gpu_memory_logging.interval_s = 2.5
    cfg.runner.gpu_memory_logging.output_dir = str(tmp_path / "explicit")
    validate_gpu_memory_logging_cfg(cfg)
    assert cfg.runner.gpu_memory_logging.interval_s == 2.5
    assert cfg.runner.gpu_memory_logging.output_dir == str(tmp_path / "explicit")
    assert len(cfg.runner.gpu_memory_logging.run_id) == 32


def test_worker_registration_separates_scheduler_and_main_pids(tmp_path):
    worker = object.__new__(Worker)
    worker._worker_address = WorkerAddress.from_name("rollout:2:0")
    worker._worker_info = SimpleNamespace(cluster_node_rank=0)
    worker._rank = 0
    worker.torch_platform = None
    options = _options(tmp_path)
    worker.configure_memory_logging(options, "rollout", "scheduler")
    worker.set_memory_stage(16, "rollout_offload/after")
    path = tmp_path / f"test_run/processes/node_0/{os.getpid()}.json"
    metadata = json.loads(path.read_text())
    assert metadata["rank"] == 2
    assert metadata["child_rank"] == 0
    assert metadata["role"] == "scheduler"
    assert metadata["create_time"] == psutil.Process().create_time()
    records = [
        json.loads(line)
        for line in next((tmp_path / "test_run/torch/node_0").glob("*.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert records[-1]["step"] == 16
    assert records[-1]["stage"] == "rollout_offload/after"


def _host_options(tmp_path):
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {"log_path": str(tmp_path)},
                "host_memory_logging": {"enabled": True},
            }
        }
    )
    validate_gpu_memory_logging_cfg(cfg)
    options = _options(tmp_path)
    options["host"] = OmegaConf.to_container(cfg.runner.host_memory_logging)
    return options


def _cgroup(tmp_path, version=2):
    proc, cgroup = tmp_path / "proc", tmp_path / "cgroup"
    proc.mkdir()
    cgroup.mkdir()
    (proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 300 kB\n")
    if version == 2:
        (cgroup / "memory.current").write_text("800000")
        (cgroup / "memory.max").write_text("1000000")
        (cgroup / "memory.stat").write_text(
            "anon 500000\nfile 200000\nshmem 60000\nkernel 90000\ninactive_file 100000\nactive_file 50000\n"
        )
        (cgroup / "memory.events").write_text("oom 1\noom_kill 1\n")
    else:
        (cgroup / "memory.usage_in_bytes").write_text("800000")
        (cgroup / "memory.limit_in_bytes").write_text("1000000")
        (cgroup / "memory.stat").write_text(
            "total_rss 500000\ntotal_cache 200000\ntotal_shmem 60000\ntotal_inactive_file 100000\ntotal_active_file 50000\n"
        )
    return proc, cgroup


@pytest.mark.parametrize("version", [1, 2])
def test_host_accounting_matches_ray_and_preserves_overlapping_shmem(tmp_path, version):
    proc, cgroup = _cgroup(tmp_path, version)
    record = HostMemorySampler(cgroup, proc).sample(0.95)
    assert record["ray_used_bytes"] == 650000
    assert record["ray_limit_bytes"] == 1000000
    assert record["ray_threshold_bytes"] == 950000
    assert record["composition"]["file_bytes"] == 200000
    assert record["composition"]["shmem_bytes"] == 60000
    assert record["composition"]["charge_minus_anon_and_file_bytes"] == 100000


def _process(proc, pid, ppid=0, start=10, pss=30, scope="/job"):
    directory = proc / str(pid)
    directory.mkdir(exist_ok=True)
    fields = ["0"] * 22
    fields[0], fields[1], fields[19], fields[21] = "S", str(ppid), str(start), "100"
    (directory / "stat").write_text(f"{pid} (worker child) " + " ".join(fields))
    (directory / "cgroup").write_text(f"0::{scope}\n")
    (directory / "smaps_rollup").write_text(
        f"1000-2000 ---p 00000000 00:00 0 [rollup]\nPss: {pss} kB\nPrivate_Clean: 5 kB\nPrivate_Dirty: 10 kB\nPss_Anon: 20 kB\nPss_File: 5 kB\nPss_Shmem: 5 kB\n"
    )


def test_process_attribution_scope_pss_growth_pid_reuse_and_partial_coverage(tmp_path):
    proc, cgroup = _cgroup(tmp_path)
    (proc / "self").mkdir()
    (proc / "self" / "mountinfo").write_text(
        f"1 0 0:1 /job {cgroup} rw - cgroup2 cgroup rw\n"
    )
    _process(proc, 100)
    _process(proc, 101, ppid=100)
    _process(proc, 102, scope="/another_job")
    registrations = tmp_path / "registrations"
    atomic_json(
        registrations / "100.json",
        {
            "pid": 100,
            "start_ticks": 10,
            "component": "critic",
            "role": "main",
            "rank": 0,
        },
    )
    options = {
        "registrations_dir": str(registrations),
        "max_processes": 10,
        "process_scan_budget_s": 10,
    }
    writer = JsonlWriter(tmp_path / "processes.jsonl", 1000000)
    sampler = ProcessMemorySampler(
        tmp_path, options, proc, HostMemorySampler(cgroup, proc)
    )
    try:
        initial = sampler.sample(writer, {"stage": "training"}, True)
        assert initial["coverage"] == "complete"
        assert initial["processes"] == 2
        assert initial["groups"]["critic"]["pss_bytes"] == 60 * 1024
        assert initial["groups"]["critic"]["rss_bytes"] > 60 * 1024
        sampler.sample(writer, {}, False)
        _process(proc, 101, ppid=100, pss=50)
        grown = sampler.sample(writer, {}, True)
        child = next(row for row in grown["largest_growth"] if row["pid"] == 101)
        assert child["growth_bytes"] == 20 * 1024
        assert child["growth_interval_s"] > 0
        _process(proc, 100, start=20)
        (proc / "101" / "smaps_rollup").unlink()
        reused = sampler.sample(writer, {}, True)
        assert reused["coverage"] == "partial"
        assert reused["groups"]["external_or_unknown"]["processes"] == 1
        assert (
            next(row for row in reused["largest_growth"] if row["pid"] == 100)[
                "growth_bytes"
            ]
            is None
        )
    finally:
        writer.close()


def _eventually(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(
        "Memory helper did not reach the expected state within timeout"
    )


@pytest.mark.parametrize(
    "entry",
    [
        "rlinf/utils/host_memory.py",
        "rlinf/scheduler/hardware/accelerators/nvidia_memory.py",
    ],
)
def test_standalone_helper_singleton_pressure_and_bounded_rss(tmp_path, entry):
    options = _host_options(tmp_path)
    options["nvml"] = False
    options["host"].update(
        {
            "interval_s": 0.02,
            "pressure_interval_s": 0.02,
            "warning_fraction": 0.000001,
            "process_interval_s": 0.05,
            "pressure_process_interval_s": 0.05,
            "max_processes": 64,
            "process_scan_budget_s": 0.01,
        }
    )
    entrypoint = str(Path(__file__).resolve().parents[2] / entry)
    helper = start_host_memory_helper(options, 0, entrypoint)
    directory = host_memory_directory(options, 0)
    try:
        assert start_host_memory_helper(options, 0, entrypoint)["pid"] == helper["pid"]
        _eventually(lambda: (directory / "pressure.json").exists())
        assert psutil.Process(helper["pid"]).memory_info().rss < 64 * 1024 * 1024
        startup = json.loads((directory / "node.jsonl").read_text().splitlines()[0])
        assert not startup["ray_imported"] and not startup["torch_imported"]
        pressure = read_json(directory / "pressure.json", 524288)
        assert pressure["event"] == "pressure"
        assert pressure["ray_used_bytes"] > 0
        atomic_json(directory / "control.json", {"stage": "critic_training", "step": 5})
        _eventually(
            lambda: read_json(directory / "latest.json")["stage"] == "critic_training"
        )
        atomic_json(directory / "control.json", {"stop": True})
        _eventually(lambda: not identity_alive(helper))
        assert (
            json.loads((directory / "node.jsonl").read_text().splitlines()[-1])[
                "record_type"
            ]
            == "monitor_stop"
        )
    finally:
        if identity_alive(helper):
            psutil.Process(helper["pid"]).terminate()


def test_helper_owner_death_grace_still_flushes_evidence(tmp_path):
    options = _host_options(tmp_path)
    directory = host_memory_directory(options, 0)
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    settings = {
        **options["host"],
        "directory": str(directory),
        "registrations_dir": str(tmp_path / "registrations"),
        "run_id": "test",
        "node_rank": 0,
        "owner": proc_identity(owner.pid),
        "nvml": False,
        "gpu_interval_s": 1,
        "interval_s": 0.02,
        "owner_dead_grace_s": 0.1,
        "max_processes": 32,
    }
    atomic_json(directory / "config.json", settings)
    entrypoint = str(Path(__file__).resolve().parents[2] / "rlinf/utils/host_memory.py")
    helper = subprocess.Popen(
        [sys.executable, "-B", entrypoint, "--config", str(directory / "config.json")]
    )
    try:
        _eventually(lambda: (directory / "latest.json").exists())
        owner.terminate()
        owner.wait(timeout=5)
        assert helper.wait(timeout=5) == 0
        assert (
            json.loads((directory / "node.jsonl").read_text().splitlines()[-1])[
                "record_type"
            ]
            == "monitor_stop"
        )
    finally:
        if owner.poll() is None:
            owner.terminate()
            owner.wait(timeout=5)
        if helper.poll() is None:
            helper.terminate()
            helper.wait(timeout=5)


def test_cpu_payload_logging_no_copy_bounded_traversal_and_no_history(tmp_path):
    import torch

    tensor = torch.empty(1000)
    size, incomplete = logical_payload_size(
        {"empty": None, "tensor": tensor, "same_tensor": tensor, "bytes": b"123"},
        host_only=True,
    )
    assert size == 4003 and not incomplete
    assert logical_payload_size([tensor] * 10000, max_objects=64)[1]
    metadata = {**_metadata(), "start_ticks": proc_identity(os.getpid())["start_ticks"]}
    recorder = TorchMemoryRecorder(_host_options(tmp_path), metadata, None)
    recorder.update_counters({"cache_bytes": 100})
    tracemalloc.start()
    try:
        for _ in range(20):
            recorder.record("warmup")
        baseline = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        for index in range(500):
            recorder.update_counters({"cache_bytes": index})
            recorder.record("sample")
        current, peak = tracemalloc.get_traced_memory()
        assert current - baseline < 256 * 1024
        assert peak - baseline < 256 * 1024
    finally:
        tracemalloc.stop()
    state = read_json(recorder._state_path)
    assert state["application_counters"]["cache_bytes"] == 499
    assert state["devices"] == []
    with pytest.raises(ValueError, match="scalar"):
        recorder.update_counters({"tensor": tensor})


def test_streaming_rotation_and_oversized_metadata_rejection(tmp_path):
    writer = JsonlWriter(tmp_path / "memory.jsonl", 512)
    for index in range(100):
        writer.write({"sample": index, "data": "x" * 100})
    writer.close()
    paths = sorted(tmp_path.glob("memory.jsonl*"))
    assert len(paths) == 3
    assert all(path.stat().st_size < 700 for path in paths)
    (tmp_path / "oversized.json").write_text("x" * 16385)
    with pytest.raises(ValueError, match="exceeds"):
        read_json(tmp_path / "oversized.json")


def test_host_only_options_and_pinned_pool_active_cached_accounting(tmp_path):
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {"log_path": str(tmp_path)},
                "host_memory_logging": {"enabled": True},
            }
        }
    )
    validate_gpu_memory_logging_cfg(cfg)
    options = memory_logging_options(cfg.runner)
    assert not options["nvml"] and not options["torch"]

    class Platform:
        def is_initialized(self):
            return True

        def host_memory_stats(self):
            return {
                "allocated_bytes.current": 300,
                "allocated_bytes.peak": 500,
                "active_bytes.current": 100,
                "active_bytes.peak": 200,
            }

    metadata = {**_metadata(), "start_ticks": proc_identity(os.getpid())["start_ticks"]}
    recorder = TorchMemoryRecorder(options, metadata, Platform())
    recorder.record("offload")
    state = read_json(recorder._state_path)
    assert state["pinned_allocator"]["allocated_bytes.current"] == 300
    assert state["pinned_allocator"]["active_bytes.current"] == 100
    assert state["devices"] == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("interval_s", 0),
        ("state_interval_s", float("nan")),
        ("warning_fraction", 1.0),
        ("max_processes", 4097),
        ("max_log_bytes", 0),
    ],
)
def test_host_config_rejects_unbounded_or_invalid_sampling(tmp_path, key, value):
    cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {"log_path": str(tmp_path)},
                "host_memory_logging": {"enabled": True, key: value},
            }
        }
    )
    with pytest.raises(ValueError, match="runner.host_memory_logging"):
        validate_gpu_memory_logging_cfg(cfg)
