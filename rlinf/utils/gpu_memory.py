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
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rlinf.scheduler import Worker
from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorManager
from rlinf.utils.logging import get_logger


def _timestamp() -> dict[str, Any]:
    return {
        "timestamp_ns": time.time_ns(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def register_memory_process(
    options: dict[str, Any], worker: Worker, component: str, role: str
) -> dict[str, Any]:
    """Publish a PID/start-time identity for attribution on this process's node."""
    import psutil

    process = psutil.Process()
    address = worker.worker_address
    metadata = {
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
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata), encoding="utf-8")
    temporary.replace(path)
    return metadata


class TorchMemoryRecorder:
    """Record the local allocator without synchronizing or initializing devices."""

    def __init__(
        self, options: dict[str, Any], metadata: dict[str, Any], platform: Any
    ) -> None:
        self._enabled = options["torch"]
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

    def record(self, event: str) -> None:
        """Append allocated/reserved and their peaks for each visible device."""
        if not self._enabled:
            return
        try:
            with self._lock:
                platform = self._platform
                initialized = platform is not None and platform.is_initialized()
                devices = []
                if initialized:
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
                _write_jsonl(
                    self._path,
                    {
                        **_timestamp(),
                        **self._metadata,
                        "record_type": "torch",
                        "step": self._step,
                        "stage": self._stage,
                        "event": event,
                        "peak_scope": self._peak_scope,
                        "device_initialized": initialized,
                        "devices": devices,
                    },
                )
        except Exception as exc:
            self._warn(exc)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Capture an operation's allocator peak, including its failure exit."""
        self.record(f"{name}/before")
        self._reset_peak(name)
        try:
            yield
        except BaseException:
            self.record(f"{name}/error")
            raise
        finally:
            self.record(f"{name}/after")


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
            command = process.cmdline()
            info.update(
                ppid=process.ppid(),
                create_time=process.create_time(),
                process_name=process.name(),
                command=command,
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
                for path in directory.glob("*.json"):
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    registrations[metadata["pid"]] = metadata
                devices = self._sampler.sample()
                identities = {}
                for device in devices:
                    for process in device["processes"]:
                        pid = process["pid"]
                        if pid not in identities:
                            identities[pid] = self._process_metadata(pid, registrations)
                        process.update(identities[pid])
                    known = sum(
                        process["used_bytes"] or 0 for process in device["processes"]
                    )
                    device["known_process_used_bytes"] = known
                    device["device_minus_known_process_bytes"] = (
                        device["used_bytes"] - known
                    )
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
