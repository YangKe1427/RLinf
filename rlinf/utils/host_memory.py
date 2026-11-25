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

"""Stream Linux memory accounting without importing Ray, PyTorch, or models.

The node helper runs this file directly so scheduler package imports do not
load accelerator libraries. Process PSS and application counters are separate
views of memory; neither is added to the cgroup's physical accounting.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import socket
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path


def atomic_json(path: Path, record: dict) -> None:
    """Publish a small control or summary record without exposing partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def read_json(path: Path, limit: int = 16384) -> dict:
    """Read a bounded metadata record, rejecting oversized or partial input."""
    with path.open(encoding="utf-8") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f"Memory metadata exceeds {limit} characters: {path}")
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError(f"Memory metadata must be a JSON object: {path}")
    return result


def proc_identity(pid: int, proc_root: Path = Path("/proc")) -> dict:
    """Return a PID/start-tick identity and RSS without walking memory mappings."""
    value = (proc_root / str(pid) / "stat").read_text()
    left, right = value.index("("), value.rindex(")")
    fields = value[right + 2 :].split()
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "ppid": int(fields[1]),
        "process_name": value[left + 1 : right][:128],
        "state": fields[0],
        "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
    }


def identity_alive(identity: dict, proc_root: Path = Path("/proc")) -> bool:
    """Check start ticks as well as PID so reused PIDs cannot keep a run alive."""
    try:
        info = proc_identity(identity["pid"], proc_root)
        return info["state"] != "Z" and info["start_ticks"] == identity["start_ticks"]
    except (OSError, ValueError, IndexError):
        return False


def proc_status(pid: int, proc_root: Path = Path("/proc")) -> dict:
    """Read cheap resident, locked, and swap counters from proc status."""
    names = {"VmRSS": "rss_bytes", "VmLck": "locked_bytes", "VmSwap": "swap_bytes"}
    result = {}
    with (proc_root / str(pid) / "status").open() as stream:
        for line in stream:
            name, _, value = line.partition(":")
            if name in names:
                result[names[name]] = int(value.split()[0]) * 1024
    return result


def proc_command(pid: int, proc_root: Path = Path("/proc")) -> dict:
    """Bound command metadata before splitting padded or oversized arguments."""
    with (proc_root / str(pid) / "cmdline").open("rb") as stream:
        raw = stream.read(4097)
    arguments = [
        argument
        for argument in raw[:4096].decode(errors="replace").split("\0")
        if argument
    ]
    return {
        "command": [argument[:256] for argument in arguments[:16]],
        "command_truncated": len(raw) > 4096
        or len(arguments) > 16
        or any(len(argument) > 256 for argument in arguments[:16]),
    }


class JsonlWriter:
    """Write one record at a time with a fixed buffer and bounded disk retention."""

    def __init__(self, path: Path, max_bytes: int, backups: int = 2) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("a", encoding="utf-8", buffering=16384)
        self._bytes = path.stat().st_size

    def write(self, record: dict, flush: bool = True) -> None:
        """Stream JSON directly; no queue or history retains caller records."""
        with self._lock:
            if self._bytes >= self.max_bytes:
                self._stream.close()
                for index in range(self.backups, 0, -1):
                    source = (
                        self.path if index == 1 else Path(f"{self.path}.{index - 1}")
                    )
                    target = Path(f"{self.path}.{index}")
                    if source.exists():
                        source.replace(target)
                self._stream = self.path.open("a", encoding="utf-8", buffering=16384)
                self._bytes = 0
            for piece in json.JSONEncoder(
                ensure_ascii=False, separators=(",", ":")
            ).iterencode(record):
                self._stream.write(piece)
                self._bytes += len(piece.encode("utf-8"))
            self._stream.write("\n")
            self._bytes += 1
            if flush:
                self._stream.flush()

    def close(self) -> None:
        """Flush and close the file owned by this writer."""
        with self._lock:
            self._stream.close()


class HostMemorySampler:
    """Read cgroup and host counters in the accounting used by Ray on Linux."""

    def __init__(
        self,
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        proc_root: Path = Path("/proc"),
    ) -> None:
        self.proc_root = proc_root
        self.cgroup_root = cgroup_root
        self.v2 = (cgroup_root / "memory.current").exists()
        if not self.v2 and (cgroup_root / "memory").is_dir():
            self.cgroup_root = cgroup_root / "memory"
        self.membership_path = "/"
        try:
            for line in (proc_root / "self" / "mountinfo").read_text().splitlines():
                fields = line.split()
                if fields[4] == str(self.cgroup_root):
                    self.membership_path = fields[3]
                    break
        except OSError:
            pass

    def contains_process(self, pid: int) -> bool:
        """Restrict attribution to the accounting cgroup and its descendants."""
        if not (self.cgroup_root / "memory.stat").exists():
            return True
        with (self.proc_root / str(pid) / "cgroup").open() as stream:
            for line in stream:
                _, controllers, path = line.strip().split(":", 2)
                if (self.v2 and not controllers) or "memory" in controllers.split(","):
                    scope = self.membership_path.rstrip("/")
                    return path == scope or path.startswith(scope + "/")
        return False

    @staticmethod
    def _integer(path: Path) -> int | None:
        try:
            value = path.read_text().strip()
            return int(value) if value != "max" else None
        except (OSError, ValueError):
            return None

    @staticmethod
    def _key_values(path: Path, kilobytes: bool = False) -> dict:
        values = {}
        try:
            with path.open() as stream:
                for line in stream:
                    fields = line.replace(":", "").split()
                    if len(fields) >= 2 and fields[1].isdecimal():
                        values[fields[0]] = int(fields[1]) * (1024 if kilobytes else 1)
        except (OSError, ValueError):
            return {}
        return values

    def sample(self, threshold: float) -> dict:
        """Preserve raw counters, cache exclusions, and unclassified charge."""
        host = self._key_values(self.proc_root / "meminfo", kilobytes=True)
        prefix = "" if self.v2 else "total_"
        stat = self._key_values(self.cgroup_root / "memory.stat")
        usage = self._integer(
            self.cgroup_root
            / ("memory.current" if self.v2 else "memory.usage_in_bytes")
        )
        limit = self._integer(
            self.cgroup_root / ("memory.max" if self.v2 else "memory.limit_in_bytes")
        )
        total = host.get("MemTotal")
        if limit is not None:
            total = min(total, limit) if total is not None else limit
        inactive = stat.get(f"{prefix}inactive_file")
        active = stat.get(f"{prefix}active_file")
        used = (
            usage - inactive - active
            if usage is not None and inactive is not None and active is not None
            else None
        )
        source = "cgroup_v2" if self.v2 else "cgroup_v1"
        if used is None:
            # Match psutil.virtual_memory().used on Linux, Ray's host fallback.
            excluded = sum(
                host.get(key, 0)
                for key in ("MemFree", "Buffers", "Cached", "SReclaimable")
            )
            used = host["MemTotal"] - excluded if "MemTotal" in host else None
            source = "host_psutil_used_fallback"
        composition = {
            "anon_bytes": stat.get(f"{prefix}{'anon' if self.v2 else 'rss'}"),
            "file_bytes": stat.get(f"{prefix}{'file' if self.v2 else 'cache'}"),
            "shmem_bytes": stat.get(f"{prefix}shmem"),
            "kernel_bytes": stat.get("kernel"),
        }
        anon, file = composition["anon_bytes"], composition["file_bytes"]
        composition["charge_minus_anon_and_file_bytes"] = (
            usage - anon - file
            if usage is not None and anon is not None and file is not None
            else None
        )
        charge_components = {
            "anon_bytes": anon,
            "file_minus_excluded_cache_bytes": file - inactive - active
            if file is not None and inactive is not None and active is not None
            else None,
            "other_charge_bytes": composition["charge_minus_anon_and_file_bytes"],
        }
        return {
            "accounting_source": source,
            "cgroup_path": str(self.cgroup_root),
            "process_cgroup_scope": self.membership_path,
            "cgroup_current_bytes": usage,
            "cgroup_limit_bytes": limit,
            "ray_used_bytes": used,
            "ray_limit_bytes": total,
            "ray_threshold_bytes": int(total * threshold) if total else None,
            "ray_usage_fraction": used / total if used is not None and total else None,
            "excluded_inactive_file_bytes": inactive,
            "excluded_active_file_bytes": active,
            "composition": composition,
            "ray_charge_components": charge_components,
            "sampling_is_non_atomic": True,
            "memory_stat": stat,
            "memory_events": self._key_values(self.cgroup_root / "memory.events"),
            "swap_current_bytes": self._integer(
                self.cgroup_root / "memory.swap.current"
            ),
            "swap_limit_bytes": self._integer(self.cgroup_root / "memory.swap.max"),
            "host_meminfo": {
                key: host.get(key)
                for key in (
                    "MemTotal",
                    "MemAvailable",
                    "Cached",
                    "Shmem",
                    "Unevictable",
                    "Mlocked",
                )
            },
            "pressure": self._pressure(),
            "composition_note": "shmem is part of file; process PSS and application bytes are separate views",
        }

    def _pressure(self) -> dict:
        result = {}
        path = self.cgroup_root / "memory.pressure"
        if not path.exists():
            path = self.proc_root / "pressure" / "memory"
        try:
            with path.open() as stream:
                for line in stream:
                    kind, *fields = line.split()
                    result[kind] = {
                        key: float(value)
                        for key, value in (field.split("=") for field in fields)
                    }
        except (OSError, ValueError):
            return {}
        return result


class ProcessMemorySampler:
    """Stream process rows and retain only bounded identities and a prior scan."""

    def __init__(
        self,
        directory: Path,
        options: dict,
        proc_root: Path = Path("/proc"),
        host: HostMemorySampler | None = None,
    ) -> None:
        self.directory = directory
        self.options = options
        self.proc_root = proc_root
        self._previous = {}
        self._identities = {}
        self._cursor = 0
        self._host = host or HostMemorySampler(proc_root=proc_root)

    def _registrations(self) -> dict:
        registrations = {}
        for index, path in enumerate(
            Path(self.options["registrations_dir"]).glob("*.json")
        ):
            if index >= self.options["max_processes"]:
                break
            try:
                metadata = read_json(path)
                registrations[metadata["pid"]] = metadata
            except (OSError, ValueError, KeyError):
                continue
        return registrations

    def _metadata(self, info: dict, registrations: dict, processes: dict) -> dict:
        ancestor = info
        for _ in range(32):
            registered = registrations.get(ancestor["pid"])
            if registered and registered.get("start_ticks") == ancestor["start_ticks"]:
                return {
                    "component": registered["component"],
                    "rank": registered.get("rank"),
                    "role": registered["role"]
                    if ancestor["pid"] == info["pid"]
                    else "child_process",
                    "attribution": "registered"
                    if ancestor["pid"] == info["pid"]
                    else "ancestor",
                }
            ancestor = processes.get(ancestor["ppid"])
            if ancestor is None:
                break
        return {
            "component": "external_or_unknown",
            "role": "unknown",
            "attribution": "unregistered",
        }

    def _rollup(self, pid: int) -> dict:
        values = HostMemorySampler._key_values(
            self.proc_root / str(pid) / "smaps_rollup", True
        )
        if "Pss" not in values:
            return {"pss_available": False}
        return {
            "pss_available": True,
            "pss_bytes": values["Pss"],
            "uss_bytes": values.get("Private_Clean", 0)
            + values.get("Private_Dirty", 0),
            "pss_anon_bytes": values.get("Pss_Anon"),
            "pss_file_bytes": values.get("Pss_File"),
            "pss_shmem_bytes": values.get("Pss_Shmem"),
            "swap_pss_bytes": values.get("SwapPss"),
            "locked_bytes": values.get("Locked"),
        }

    def sample(self, writer: JsonlWriter, context: dict, detailed: bool) -> dict:
        """Write rows as they are read and expose coverage and collection age."""
        started = time.monotonic()
        timestamp = time.time_ns()
        registrations = self._registrations()
        processes = {}
        skipped = 0
        for path in self.proc_root.iterdir():
            if not path.name.isdecimal():
                continue
            if len(processes) >= self.options["max_processes"]:
                skipped += 1
                continue
            try:
                if not self._host.contains_process(int(path.name)):
                    continue
                info = proc_identity(int(path.name), self.proc_root)
                processes[info["pid"]] = info
            except (OSError, ValueError, IndexError):
                skipped += 1
        groups = {}
        growth = []
        previous = {}
        identities = {}
        pss_count = 0
        pid_list = sorted(processes)
        if detailed and pid_list:
            offset = self._cursor % len(pid_list)
            pid_list = pid_list[offset:] + pid_list[:offset]
        budget_exhausted = False
        for index, pid in enumerate(pid_list):
            info = processes[pid]
            identity = (pid, info["start_ticks"])
            metadata = self._metadata(info, registrations, processes)
            if (
                metadata["attribution"] == "unregistered"
                and identity in self._identities
            ):
                metadata = self._identities[identity]
            identities[identity] = metadata
            if identity not in self._identities:
                try:
                    command = proc_command(pid, self.proc_root)
                    writer.write(
                        {
                            **context,
                            **info,
                            **metadata,
                            **command,
                            "uid": (self.proc_root / str(pid)).stat().st_uid,
                            "record_type": "host_process_metadata",
                            "timestamp_ns": time.time_ns(),
                        },
                        flush=False,
                    )
                except OSError:
                    pass
            row = {
                **context,
                **info,
                **metadata,
                "record_type": "host_process",
                "timestamp_ns": time.time_ns(),
            }
            if detailed and not budget_exhausted:
                row.update(self._rollup(pid))
                row["pss_timestamp_ns"] = time.time_ns()
                pss_count += int(row["pss_available"])
                budget_exhausted = (
                    time.monotonic() - started >= self.options["process_scan_budget_s"]
                )
                if budget_exhausted:
                    self._cursor += index + 1
            try:
                state = read_json(self.directory / "process_state" / f"{pid}.json")
                if state.get("start_ticks") == info["start_ticks"]:
                    row["application"] = state
                    row["application_age_s"] = max(
                        0, (time.time_ns() - state["timestamp_ns"]) / 1e9
                    )
            except (OSError, ValueError, KeyError):
                pass
            try:
                if (
                    proc_identity(pid, self.proc_root)["start_ticks"]
                    != info["start_ticks"]
                ):
                    skipped += 1
                    continue
            except (OSError, ValueError, IndexError):
                skipped += 1
                continue
            component = metadata["component"]
            if component not in groups and len(groups) >= 64:
                component = "other_components"
            group = groups.setdefault(
                component,
                {"rss_bytes": 0, "pss_bytes": 0, "processes": 0, "pss_processes": 0},
            )
            group["processes"] += 1
            group["rss_bytes"] += info["rss_bytes"]
            group["pss_bytes"] += row.get("pss_bytes", 0)
            group["pss_processes"] += int(row.get("pss_available", False))
            metric = "pss_bytes" if row.get("pss_available") else "rss_bytes"
            value = row[metric]
            old = self._previous.get(identity, {})
            delta = value - old[metric] if metric in old else None
            row["growth_bytes"] = delta
            row["growth_metric"] = metric
            prior_timestamp = old.get(f"{metric}_timestamp_ns")
            row["growth_interval_s"] = (
                (row["timestamp_ns"] - prior_timestamp) / 1e9
                if prior_timestamp
                else None
            )
            previous[identity] = {
                "rss_bytes": info["rss_bytes"],
                "rss_bytes_timestamp_ns": row["timestamp_ns"],
            }
            if "pss_bytes" in row:
                previous[identity]["pss_bytes"] = row["pss_bytes"]
                previous[identity]["pss_bytes_timestamp_ns"] = row["pss_timestamp_ns"]
            elif "pss_bytes" in old:
                previous[identity]["pss_bytes"] = old["pss_bytes"]
                previous[identity]["pss_bytes_timestamp_ns"] = old[
                    "pss_bytes_timestamp_ns"
                ]
            rank = delta if delta is not None else value
            entry = (
                rank,
                pid,
                {
                    key: row[key]
                    for key in (
                        "pid",
                        "start_ticks",
                        "process_name",
                        "component",
                        "rss_bytes",
                        "growth_metric",
                        "growth_bytes",
                    )
                },
            )
            if "pss_bytes" in row:
                entry[2]["pss_bytes"] = row["pss_bytes"]
            entry[2]["growth_interval_s"] = row["growth_interval_s"]
            if "application" in row:
                state = row["application"]
                entry[2]["application"] = {
                    key: state[key]
                    for key in (
                        "stage",
                        "phase",
                        "operation_index",
                        "application_counters",
                        "pinned_allocator",
                        "counter_timestamp_ns",
                    )
                    if key in state
                }
                entry[2]["application_age_s"] = row["application_age_s"]
            if len(growth) < 20:
                heapq.heappush(growth, entry)
            elif rank > growth[0][0]:
                heapq.heapreplace(growth, entry)
            writer.write(row, flush=False)
        self._previous, self._identities = previous, identities
        summary = {
            **context,
            "record_type": "host_process_summary",
            "timestamp_ns": timestamp,
            "scan_duration_s": time.monotonic() - started,
            "processes": len(processes),
            "skipped_processes": skipped,
            "pss_processes": pss_count,
            "pss_requested": detailed,
            "scan_budget_exhausted": budget_exhausted,
            "groups": groups,
            "largest_growth": [entry[2] for entry in sorted(growth, reverse=True)],
            "coverage": "complete"
            if detailed and pss_count == len(processes) and not skipped
            else "partial",
            "note": "RSS repeats shared pages; partial PSS totals do not explain the full cgroup charge",
        }
        writer.write(summary)
        return summary


def ray_threshold(proc_root: Path = Path("/proc")) -> tuple[float, str]:
    """Use the running raylet's startup threshold, with an explicit fallback."""
    for path in proc_root.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            if (path / "comm").read_text().strip() != "raylet":
                continue
            with (path / "cmdline").open("rb") as stream:
                arguments = stream.read(65536).decode(errors="replace").split("\0")
            logfile = next(
                (
                    arg.split("=", 1)[1]
                    for arg in arguments
                    if arg.startswith("--stdout_filepath=")
                ),
                None,
            )
            if logfile:
                with open(logfile) as stream:
                    header = stream.read(131072)
                match = re.search(r"usage threshold at \d+ bytes \(([\d.]+)%", header)
                if match:
                    return float(match[1]) / 100, "raylet_startup_log"
            with (path / "environ").open("rb") as stream:
                environment = stream.read(262144).split(b"\0")
            for item in environment:
                if item.startswith(b"RAY_memory_usage_threshold="):
                    return float(item.split(b"=", 1)[1]), "raylet_environment"
        except (OSError, ValueError):
            continue
    return 0.95, "ray_default_unverified"


class NodeHostMemoryMonitor:
    """Own independent node, process, and optional NVML sampling loops."""

    def __init__(self, options: dict, sampler_factory: Callable | None = None) -> None:
        self.options = options
        self.directory = Path(options["directory"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._sampler = HostMemorySampler()
        self._processes = ProcessMemorySampler(
            self.directory, options, host=self._sampler
        )
        self._threshold, self._threshold_source = ray_threshold()
        self._context = {
            "run_id": options["run_id"],
            "node_rank": options["node_rank"],
            "step": None,
            "stage": "initialization",
        }
        self._process_summary = {}
        self._fraction = 0.0
        self._threads = []
        self._baseline = None
        self._last_pressure = 0.0
        self._owner_dead_at = None
        self._sampler_factory = sampler_factory

    def _writer(self, name: str) -> JsonlWriter:
        return JsonlWriter(self.directory / name, self.options["max_log_bytes"])

    def _process_loop(self) -> None:
        writer = self._writer("processes.jsonl")
        last_detail = 0.0
        try:
            while not self._stop.is_set():
                interval = (
                    self.options["pressure_process_interval_s"]
                    if self._fraction >= self.options["warning_fraction"]
                    else self.options["process_interval_s"]
                )
                detail_interval = (
                    self.options["pressure_detail_interval_s"]
                    if self._fraction >= self.options["warning_fraction"]
                    else self.options["detail_interval_s"]
                )
                detailed = time.monotonic() - last_detail >= detail_interval
                try:
                    self._process_summary = self._processes.sample(
                        writer, dict(self._context), detailed
                    )
                    if detailed:
                        last_detail = time.monotonic()
                except (OSError, ValueError) as error:
                    writer.write(
                        {
                            **self._context,
                            "record_type": "monitor_error",
                            "timestamp_ns": time.time_ns(),
                            "error": str(error)[:512],
                        }
                    )
                self._stop.wait(interval)
        finally:
            writer.close()

    def _nvml_loop(self) -> None:
        writer = self._writer("nvml.jsonl")
        sampler = None
        try:
            sampler = self._sampler_factory()
            while not self._stop.is_set():
                devices = sampler.sample()
                for device in devices:
                    for process in device["processes"]:
                        try:
                            info = proc_identity(process["pid"])
                            process.update(
                                {
                                    key: info[key]
                                    for key in ("start_ticks", "process_name")
                                }
                            )
                        except (OSError, ValueError, IndexError) as error:
                            process["identity_error"] = str(error)[:128]
                    known = sum(
                        process["used_bytes"] or 0 for process in device["processes"]
                    )
                    device["known_process_used_bytes"] = known
                    device["device_minus_known_process_bytes"] = (
                        device["used_bytes"] - known
                    )
                writer.write(
                    {
                        **self._context,
                        "timestamp_ns": time.time_ns(),
                        "record_type": "nvml",
                        "devices": devices,
                    }
                )
                self._stop.wait(self.options["gpu_interval_s"])
        except Exception as error:
            # Vendor telemetry failure must not stop host pressure sampling.
            writer.write(
                {
                    "record_type": "monitor_error",
                    "error": str(error)[:512],
                    "timestamp_ns": time.time_ns(),
                }
            )
        finally:
            if sampler is not None:
                sampler.close()
            writer.close()

    def run(self) -> None:
        """Persist pressure evidence before worker death, without Ray RPCs."""
        atomic_json(
            Path(self.options["registrations_dir"]) / f"{os.getpid()}.json",
            {
                **proc_identity(os.getpid()),
                "component": "memory_monitor",
                "role": "standalone",
                "rank": self.options["node_rank"],
            },
        )
        writer = self._writer("node.jsonl")
        writer.write(
            {
                **self._context,
                "record_type": "monitor_start",
                "timestamp_ns": time.time_ns(),
                "ray_imported": "ray" in sys.modules,
                "torch_imported": "torch" in sys.modules,
                "options": {
                    key: self.options[key]
                    for key in (
                        "interval_s",
                        "pressure_interval_s",
                        "process_interval_s",
                        "detail_interval_s",
                        "max_processes",
                        "max_log_bytes",
                        "process_scan_budget_s",
                    )
                },
            }
        )
        for target in (self._process_loop,):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        if self.options["nvml"] and self._sampler_factory is not None:
            thread = threading.Thread(target=self._nvml_loop, daemon=True)
            thread.start()
            self._threads.append(thread)
        try:
            while not self._stop.is_set():
                try:
                    control = read_json(self.directory / "control.json")
                    if control.get("stop"):
                        break
                    self._context.update(
                        {
                            key: control[key]
                            for key in ("step", "stage")
                            if key in control
                        }
                    )
                except (OSError, ValueError):
                    pass
                if not identity_alive(self.options["owner"]):
                    if self._owner_dead_at is None:
                        self._owner_dead_at = time.monotonic()
                    if (
                        time.monotonic() - self._owner_dead_at
                        >= self.options["owner_dead_grace_s"]
                    ):
                        break
                record = {
                    **self._context,
                    **self._sampler.sample(self._threshold),
                    "record_type": "host_node",
                    "timestamp_ns": time.time_ns(),
                    "hostname": socket.gethostname(),
                    "threshold_source": self._threshold_source,
                    "monitor_rss_bytes": proc_status(os.getpid()).get("rss_bytes"),
                }
                self._fraction = record["ray_usage_fraction"] or 0
                previous = self._baseline
                record["growth_since_below_warning"] = {
                    key: record["ray_charge_components"][key]
                    - previous["ray_charge_components"][key]
                    for key in record["ray_charge_components"]
                    if previous
                    and record["ray_charge_components"][key] is not None
                    and previous["ray_charge_components"][key] is not None
                }
                record["growth_baseline_timestamp_ns"] = (
                    previous["timestamp_ns"] if previous else None
                )
                record["ray_threshold_exceeded"] = self._fraction >= self._threshold
                if (
                    previous
                    and record["ray_used_bytes"] is not None
                    and previous["ray_used_bytes"] is not None
                ):
                    record["ray_used_growth_bytes"] = (
                        record["ray_used_bytes"] - previous["ray_used_bytes"]
                    )
                if self._fraction < self.options["warning_fraction"]:
                    self._baseline = record
                pressure = self._fraction >= self.options["warning_fraction"]
                record["event"] = "pressure" if pressure else "periodic"
                writer.write(record)
                if (
                    pressure
                    and time.monotonic() - self._last_pressure
                    >= self.options["pressure_summary_interval_s"]
                ):
                    summary = {
                        **record,
                        "process_summary": self._process_summary,
                        "process_summary_age_s": (
                            time.time_ns() - self._process_summary["timestamp_ns"]
                        )
                        / 1e9
                        if self._process_summary
                        else None,
                    }
                    atomic_json(self.directory / "pressure.json", summary)
                    self._last_pressure = time.monotonic()
                atomic_json(
                    self.directory / "latest.json",
                    {
                        key: value
                        for key, value in record.items()
                        if key not in ("memory_stat", "host_meminfo", "pressure")
                    },
                )
                self._stop.wait(
                    self.options["pressure_interval_s"]
                    if pressure
                    else self.options["interval_s"]
                )
        finally:
            self._stop.set()
            for thread in self._threads:
                thread.join(timeout=1)
            writer.write(
                {
                    **self._context,
                    "record_type": "monitor_stop",
                    "timestamp_ns": time.time_ns(),
                }
            )
            writer.close()


def main(sampler_factory: Callable | None = None) -> None:
    """Run the helper in its own process group with no scheduler imports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    NodeHostMemoryMonitor(read_json(arguments.config), sampler_factory).run()


if __name__ == "__main__":
    main()
