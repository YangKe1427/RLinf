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
from types import SimpleNamespace

import psutil
import pytest
from omegaconf import OmegaConf

from rlinf.config import validate_gpu_memory_logging_cfg
from rlinf.scheduler import AcceleratorType, Worker, WorkerAddress
from rlinf.scheduler.hardware.accelerators.nvidia_memory import NvidiaMemorySampler
from rlinf.utils.gpu_memory import DeviceMemoryMonitor, TorchMemoryRecorder


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
    import ray._private.thirdparty.pynvml as nvml

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
