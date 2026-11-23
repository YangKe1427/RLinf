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

"""Physical NVIDIA device and process memory snapshots without CUDA calls."""

from typing import Any


class NvidiaMemorySampler:
    """Own an NVML session and include compute and EGL/graphics processes."""

    def __init__(self) -> None:
        """Open the Ray-bundled NVML session lazily."""
        import ray._private.thirdparty.pynvml as nvml

        self._nvml = nvml
        nvml.nvmlInit()

    def sample(self) -> list[dict[str, Any]]:
        """Return bytes per physical GPU and PID, keeping unavailable bytes null."""
        nvml = self._nvml
        devices = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            uuid = nvml.nvmlDeviceGetUUID(handle)
            pci = nvml.nvmlDeviceGetPciInfo(handle).busId
            processes = {}
            errors = []
            for kind, query in (
                ("compute", nvml.nvmlDeviceGetComputeRunningProcesses),
                ("graphics", nvml.nvmlDeviceGetGraphicsRunningProcesses),
            ):
                try:
                    entries = query(handle)
                except nvml.NVMLError as exc:
                    errors.append(f"{kind}: {exc}")
                    continue
                for entry in entries:
                    used = entry.usedGpuMemory
                    if used is None or used < 0 or used == (1 << 64) - 1:
                        used = None
                    else:
                        used = int(used)
                    process = processes.setdefault(
                        entry.pid,
                        {"pid": entry.pid, "used_bytes": None, "kinds": []},
                    )
                    process["kinds"].append(kind)
                    # A PID can appear in both lists; its memory is one allocation.
                    if used is not None:
                        process["used_bytes"] = max(process["used_bytes"] or 0, used)
            devices.append(
                {
                    "gpu_index": index,
                    "gpu_uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
                    "pci_bus_id": pci.decode() if isinstance(pci, bytes) else pci,
                    "total_bytes": int(memory.total),
                    "used_bytes": int(memory.used),
                    "free_bytes": int(memory.free),
                    "processes": list(processes.values()),
                    "process_query_errors": errors,
                }
            )
        return devices

    def close(self) -> None:
        """Release this sampler's NVML session."""
        self._nvml.nvmlShutdown()
