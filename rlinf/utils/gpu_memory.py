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

"""Stage-tagged node memory sampling and process-local allocator records."""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rlinf.scheduler import Worker
from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorManager
from rlinf.utils.host_memory import (
    atomic_json,
    identity_alive,
    proc_command,
    proc_identity,
    proc_status,
    read_json,
)
from rlinf.utils.logging import get_logger


def _timestamp() -> dict[str, Any]:
    return {
        "timestamp_ns": time.time_ns(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= 64 * 1024 * 1024:
        for index in (2, 1):
            source = path if index == 1 else Path(f"{path}.1")
            if source.exists():
                source.replace(Path(f"{path}.{index}"))
    with path.open("a", encoding="utf-8") as stream:
        for piece in json.JSONEncoder(
            ensure_ascii=False, separators=(",", ":")
        ).iterencode(record):
            stream.write(piece)
        stream.write("\n")


def host_memory_directory(options: dict, node_rank: int) -> Path:
    """Return the node's control and host evidence directory."""
    return (
        Path(options["output_dir"]) / options["run_id"] / "host" / f"node_{node_rank}"
    )


def start_host_memory_helper(options: dict, node_rank: int, entrypoint: str) -> dict:
    """Start at most one detached, lightweight helper per run and node."""
    import fcntl

    directory = host_memory_directory(options, node_rank)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "startup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            existing = read_json(directory / "helper.json")
            if identity_alive(existing):
                return existing
        except (OSError, ValueError, KeyError):
            pass
        settings = {
            **options["host"],
            "directory": str(directory),
            "registrations_dir": str(
                Path(options["output_dir"])
                / options["run_id"]
                / "processes"
                / f"node_{node_rank}"
            ),
            "run_id": options["run_id"],
            "node_rank": node_rank,
            "owner": proc_identity(os.getpid()),
            "nvml": options["nvml"],
            "gpu_interval_s": options["interval_s"],
        }
        atomic_json(directory / "config.json", settings)
        atomic_json(
            directory / "control.json", {"stage": "initialization", "stop": False}
        )
        environment = os.environ.copy()
        root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = (
            root + os.pathsep + environment.get("PYTHONPATH", "")
        )
        with (directory / "helper.stderr").open("ab") as errors:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    entrypoint,
                    "--config",
                    str(directory / "config.json"),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errors,
                start_new_session=True,
                env=environment,
            )
        identity = proc_identity(child.pid)
        atomic_json(directory / "helper.json", identity)
        # Reap the child without retaining sample records or a polling queue.
        threading.Thread(
            target=child.wait, daemon=True, name="memory-helper-reaper"
        ).start()
        return identity


def logical_payload_size(
    value: Any, max_objects: int = 1024, *, host_only: bool = False
) -> tuple[int, bool]:
    """Estimate logical tensor/byte payload, with bounded traversal and no copies.

    Views can share storage and Python/native allocator overhead is excluded.
    The returned flag identifies incomplete coverage; file references are not
    charged as resident image data.
    """
    stack, seen, size, visited = [iter((value,))], set(), 0, 0
    incomplete = False
    sentinel = object()
    while stack and visited < max_objects:
        item = next(stack[-1], sentinel)
        if item is sentinel:
            stack.pop()
            continue
        visited += 1
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(item, (bytes, bytearray, memoryview)):
            size += item.nbytes if isinstance(item, memoryview) else len(item)
        elif isinstance(item, str):
            size += sys.getsizeof(item)
        elif type(item).__module__.split(".")[0] in ("torch", "numpy") and hasattr(
            item, "nbytes"
        ):
            if (
                not host_only
                or type(item).__module__.split(".")[0] == "numpy"
                or item.device.type == "cpu"
            ):
                size += int(item.nbytes)
        elif isinstance(item, (dict, tuple, list)) or (
            is_dataclass(item) and not isinstance(item, type)
        ):
            if len(stack) >= 16:
                incomplete = True
                continue
            if isinstance(item, dict):
                stack.append(iter(item.values()))
            elif isinstance(item, (tuple, list)):
                stack.append(iter(item))
            else:
                stack.append(
                    map(lambda field, obj=item: getattr(obj, field.name), fields(item))
                )
        elif item is not None and not isinstance(item, (int, float, bool)):
            incomplete = True
    return size, bool(stack) or incomplete


def report_memory_counters(
    counters: dict[str, int | float | bool | None], additive: bool = False
) -> None:
    """Update scalar holdings for a utility executing inside a Worker."""
    if Worker.current_worker is not None:
        Worker.current_worker.update_memory_counters(counters, additive)


def register_memory_process(
    options: dict[str, Any], worker: Worker, component: str, role: str
) -> dict[str, Any]:
    """Publish a PID/start-time identity for attribution on this process's node."""
    import psutil

    process = psutil.Process()
    address = worker.worker_address
    metadata = {
        "start_ticks": proc_identity(process.pid)["start_ticks"],
        "run_id": options["run_id"],
        "pid": process.pid,
        "ppid": process.ppid(),
        "create_time": process.create_time(),
        "node_rank": worker.worker_info.cluster_node_rank,
        "component": component,
        "role": role,
        "worker_address": address.get_name(),
        "rank": address.rank_path[0] if address.rank_path else worker._rank,
        "child_rank": worker._rank if len(address.rank_path) > 1 else None,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    directory = (
        Path(options["output_dir"])
        / options["run_id"]
        / "processes"
        / f"node_{metadata['node_rank']}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{process.pid}.json"
    atomic_json(path, metadata)
    return metadata


class TorchMemoryRecorder:
    """Record the local allocator without synchronizing or initializing devices."""

    def __init__(
        self, options: dict[str, Any], metadata: dict[str, Any], platform: Any
    ) -> None:
        self._enabled = options["torch"]
        self._host_enabled = options.get("host", {}).get("enabled", False)
        self._counters = {}
        self._counter_timestamp_ns = None
        self._operation_index = 0
        self._phase = None
        self._last_publish = 0.0
        self._publish_interval_s = options.get("host", {}).get("state_interval_s", 1.0)
        self._state_path = (
            host_memory_directory(options, metadata["node_rank"])
            / "process_state"
            / f"{metadata['pid']}.json"
        )
        self._metadata = metadata
        self._platform = platform
        self._step = None
        self._stage = "initialization"
        self._peak_scope = "process_start"
        self._lock = threading.RLock()
        self._warned = False
        self._path = (
            Path(options["output_dir"])
            / options["run_id"]
            / "torch"
            / f"node_{metadata['node_rank']}"
            / f"{metadata['component']}_rank_{metadata['rank']}_{metadata['role']}_pid_{metadata['pid']}.jsonl"
        )

    def update_counters(
        self, counters: dict[str, int | float | bool | None], additive: bool = False
    ) -> None:
        """Keep at most 64 scalar counters and publish at a bounded frequency."""
        if not self._host_enabled:
            return
        with self._lock:
            for key, value in counters.items():
                if len(key) > 96 or type(value) not in (int, float, bool, type(None)):
                    raise ValueError(
                        "Memory counters require keys <=96 characters and scalar values"
                    )
                if key not in self._counters and len(self._counters) >= 64:
                    raise ValueError(
                        "Memory logging supports at most 64 application counters per process"
                    )
                self._counters[key] = (
                    self._counters.get(key, 0) + value if additive else value
                )
            self._counter_timestamp_ns = time.time_ns()
            if time.monotonic() - self._last_publish >= self._publish_interval_s:
                self.record("application_update", write_event=False)

    def set_stage(self, step: int, stage: str, reset_peak: bool = False) -> None:
        """Associate future local records with the runner's one-based step."""
        with self._lock:
            self._step, self._stage = step, stage
            self.record("stage_boundary")
            if reset_peak:
                self._reset_peak(stage)

    def _reset_peak(self, scope: str) -> None:
        if not self._enabled:
            return
        try:
            if self._platform is not None and self._platform.is_initialized():
                for device in range(self._platform.device_count()):
                    self._platform.reset_peak_memory_stats(device)
            self._peak_scope = scope
        except Exception as exc:
            self._warn(exc)

    def _warn(self, exc: Exception) -> None:
        if not self._warned:
            get_logger().warning("PyTorch memory logging failed: %s", exc)
            self._warned = True

    def record(self, event: str, write_event: bool = True) -> None:
        """Append allocated/reserved and their peaks for each visible device."""
        if not self._enabled and not self._host_enabled:
            return
        try:
            with self._lock:
                platform = self._platform
                initialized = platform is not None and platform.is_initialized()
                devices = []
                if initialized and self._enabled:
                    for device in range(platform.device_count()):
                        allocated = int(platform.memory_allocated(device))
                        reserved = int(platform.memory_reserved(device))
                        devices.append(
                            {
                                "logical_device": device,
                                "allocated_bytes": allocated,
                                "reserved_bytes": reserved,
                                "reserved_minus_allocated_bytes": reserved - allocated,
                                "max_allocated_bytes": int(
                                    platform.max_memory_allocated(device)
                                ),
                                "max_reserved_bytes": int(
                                    platform.max_memory_reserved(device)
                                ),
                            }
                        )
                record = {
                    **_timestamp(),
                    **self._metadata,
                    "record_type": "torch",
                    "step": self._step,
                    "stage": self._stage,
                    "event": event,
                    "peak_scope": self._peak_scope,
                    "device_initialized": initialized,
                    "devices": devices,
                    "phase": self._phase,
                    "operation_index": self._operation_index,
                }
                if self._host_enabled:
                    record["host"] = proc_status(os.getpid())
                    record["application_counters"] = dict(self._counters)
                    record["counter_timestamp_ns"] = self._counter_timestamp_ns
                    record["application_note"] = (
                        "logical bytes exclude allocator overhead and can overlap PSS, pinned pools, and shared storage"
                    )
                    stats = (
                        getattr(platform, "host_memory_stats", None)
                        if initialized
                        else None
                    )
                    if stats is not None:
                        try:
                            values = stats()
                            record["pinned_allocator"] = {
                                key: int(values[key])
                                for key in (
                                    "allocated_bytes.current",
                                    "allocated_bytes.peak",
                                    "active_bytes.current",
                                    "active_bytes.peak",
                                )
                                if key in values
                            }
                            record["pinned_allocator_note"] = (
                                "allocated includes active and cached; native cudaMallocHost/mmap backups may be untracked"
                            )
                        except Exception as error:
                            record["pinned_allocator_error"] = str(error)[:256]
                    else:
                        record["pinned_allocator"] = {"available": False}
                    atomic_json(self._state_path, record)
                    self._last_publish = time.monotonic()
                if write_event:
                    _write_jsonl(self._path, record)
        except Exception as exc:
            self._warn(exc)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Capture an operation's allocator peak, including its failure exit."""
        with self._lock:
            previous_phase = self._phase
            self._phase = name
            self._operation_index += 1
        self.record(f"{name}/before")
        self._reset_peak(name)
        try:
            yield
        except BaseException:
            self.record(f"{name}/error")
            raise
        finally:
            self.record(f"{name}/after")
            with self._lock:
                self._phase = previous_phase


class DeviceMemoryMonitor:
    """Own one periodic physical-device sampler and JSONL writer per node."""

    def __init__(
        self, options: dict[str, Any], node_rank: int, accelerator_type: Any
    ) -> None:
        """Prepare a node-local writer and sampling lifecycle from validated config."""
        self._options = options
        self._accelerator_type = accelerator_type
        self._sampler = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._step = None
        self._stage = "initialization"
        self._warned = False
        self._directory = Path(options["output_dir"]) / options["run_id"]
        self._node_rank = node_rank
        self._path = (
            self._directory / "nvml" / f"node_{self._node_rank}_pid_{os.getpid()}.jsonl"
        )
        self._process_cache = {}

    def start(self) -> None:
        """Start the node sampler; a telemetry failure is logged, not raised."""
        try:
            manager = AcceleratorManager.manager_register.get(self._accelerator_type)
            self._sampler = manager.create_memory_sampler() if manager else None
            if self._sampler is None:
                raise RuntimeError(
                    "Physical memory sampling is unavailable on this node"
                )
        except Exception as exc:
            self._report_error(exc)
            return
        self.snapshot("monitor_start")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._options["interval_s"]):
            self.snapshot("periodic")

    def _report_error(self, exc: Exception) -> None:
        if not self._warned:
            get_logger().warning("GPU memory monitor failed: %s", exc)
            self._warned = True
        try:
            _write_jsonl(
                self._path,
                {
                    **_timestamp(),
                    "run_id": self._options["run_id"],
                    "node_rank": self._node_rank,
                    "record_type": "monitor_error",
                    "step": self._step,
                    "stage": self._stage,
                    "error": str(exc),
                },
            )
        except OSError:
            pass

    def set_stage(self, step: int, stage: str) -> None:
        """Switch the global stage and immediately snapshot all node GPUs."""
        with self._lock:
            self._step, self._stage = step, stage
            self.snapshot("stage_boundary")

    def _process_metadata(self, pid: int, registrations: dict) -> dict[str, Any]:
        import psutil

        info = {"component": "external_or_unknown", "role": "unknown"}
        try:
            process = psutil.Process(pid)
            command_info = proc_command(pid)
            command = command_info["command"]
            info.update(
                ppid=process.ppid(),
                create_time=process.create_time(),
                process_name=process.name(),
                command=command,
                command_truncated=command_info["command_truncated"],
                start_ticks=proc_identity(pid)["start_ticks"],
            )
            for ancestor in [process, *process.parents()]:
                registration = registrations.get(ancestor.pid)
                if (
                    registration
                    and registration["create_time"] == ancestor.create_time()
                ):
                    info.update(
                        {
                            key: registration[key]
                            for key in (
                                "component",
                                "role",
                                "rank",
                                "child_rank",
                                "worker_address",
                            )
                        }
                    )
                    info["attribution"] = (
                        "registered" if ancestor.pid == pid else "ancestor"
                    )
                    if ancestor.pid != pid:
                        info["role"] = "child_process"
                    break
            command_text = " ".join(command).lower()
            if "env_server.py" in command_text and "libero" in command_text:
                info["role"] = "libero_env"
            elif any(
                "sam3" in Path(argument).name.lower()
                for argument in command
                if not argument.startswith(("-", "http://", "https://"))
            ):
                info["role"] = "sam3_service"
        except (psutil.Error, OSError) as exc:
            info["process_metadata_error"] = str(exc)
        return info

    def snapshot(self, event: str = "manual") -> None:
        """Append a physical snapshot, preserving unknown PIDs and query errors."""
        if self._sampler is None:
            return
        with self._lock:
            try:
                registrations = {}
                directory = self._directory / "processes" / f"node_{self._node_rank}"
                for index, path in enumerate(directory.glob("*.json")):
                    if index >= 4096:
                        break
                    try:
                        metadata = read_json(path)
                        registrations[metadata["pid"]] = metadata
                    except (OSError, ValueError, KeyError):
                        continue
                devices = self._sampler.sample()
                identities = {}
                for device in devices:
                    for process in device["processes"]:
                        pid = process["pid"]
                        if pid not in identities:
                            identities[pid] = self._process_metadata(pid, registrations)
                            if identities[pid] != self._process_cache.get(pid):
                                _write_jsonl(
                                    self._path,
                                    {
                                        **_timestamp(),
                                        "record_type": "gpu_process_metadata",
                                        "event": "process_metadata",
                                        "run_id": self._options["run_id"],
                                        "node_rank": self._node_rank,
                                        "pid": pid,
                                        **identities[pid],
                                    },
                                )
                        process.update(
                            {
                                key: value
                                for key, value in identities[pid].items()
                                if key not in ("command", "command_truncated")
                            }
                        )
                    known = sum(
                        process["used_bytes"] or 0 for process in device["processes"]
                    )
                    device["known_process_used_bytes"] = known
                    device["device_minus_known_process_bytes"] = (
                        device["used_bytes"] - known
                    )
                self._process_cache = dict(list(identities.items())[:4096])
                _write_jsonl(
                    self._path,
                    {
                        **_timestamp(),
                        "record_type": "nvml",
                        "run_id": self._options["run_id"],
                        "node_rank": self._node_rank,
                        "step": self._step,
                        "stage": self._stage,
                        "event": event,
                        "devices": devices,
                    },
                )
            except Exception as exc:
                self._report_error(exc)

    def stop(self) -> None:
        """Flush the final sample and join the sampling thread before shutdown."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                get_logger().warning("GPU memory sampler did not stop within 5 seconds")
                return
        self.snapshot("monitor_stop")
        if self._sampler is not None:
            try:
                self._sampler.close()
            except Exception as exc:
                self._report_error(exc)
            self._sampler = None


class NodeMemoryMonitor(Worker):
    """Place a device monitor on a node through the existing Worker mechanism."""

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__()
        self._monitor = DeviceMemoryMonitor(
            options, self.worker_info.cluster_node_rank, self.accelerator_type
        )

    def start(self) -> None:
        """Start periodic device sampling on this node."""
        self._monitor.start()

    def set_stage(self, step: int, stage: str) -> None:
        """Record a stage boundary and change the periodic sample label."""
        self._monitor.set_stage(step, stage)

    def stop(self) -> None:
        """Flush and stop this node's sampler."""
        self._monitor.stop()
