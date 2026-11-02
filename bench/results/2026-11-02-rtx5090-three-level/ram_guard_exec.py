#!/usr/bin/env python3
"""Run one engine with a soft RSS/swap watchdog; this is not a hard RAM cap."""

import json
import os
from pathlib import Path
import signal
import sys
import time


GIB = 1024 ** 3


def process_sample(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rfind(")") + 2:].split()
    result = {"state": fields[0], "start_ticks": int(fields[19])}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmRSS", "VmHWM", "VmSwap"):
            result[name] = int(value.split()[0]) * 1024
    return result


def system_sample():
    available = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available = int(line.split()[1]) * 1024
            break
    if available is None:
        raise RuntimeError("MemAvailable missing")
    for line in Path("/proc/vmstat").read_text().splitlines():
        if line.startswith("pswpout "):
            return available, int(line.split()[1])
    raise RuntimeError("pswpout missing")


def write_log(path, record):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def guardian(pid, pgid, start_ticks, log_path, limits, ready_fd):
    # Staying in this private group pins its ID until the final group signal.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    null_fd = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null_fd, fd)
    if null_fd > 2:
        os.close(null_fd)
    record = {
        "guard_kind": "soft_watchdog", "hard_cap": False,
        "target_ram_bytes": 50 * GIB,
        "engine_pid": pid, "engine_pgid": pgid, "guardian_pid": os.getpid(),
        "limits": limits,
        "peaks": {"rss_bytes": 0, "hwm_bytes": 0, "swap_bytes": 0},
        "system": {"min_available_bytes": None, "pswpout_start_pages": None,
                   "pswpout_end_pages": None, "pswpout_delta_pages": 0},
        "termination": {"reason": None, "signal_sent": None},
        "samples": 0, "started_unix": time.time(), "status": "running",
    }

    def sample():
        process = process_sample(pid)
        available, pswpout = system_sample()
        if process["start_ticks"] != start_ticks:
            return "pid_reused"
        record["samples"] += 1
        for metric, field in (("rss_bytes", "VmRSS"), ("hwm_bytes", "VmHWM"),
                              ("swap_bytes", "VmSwap")):
            record["peaks"][metric] = max(record["peaks"][metric], process.get(field, 0))
        system = record["system"]
        if system["min_available_bytes"] is None:
            system["min_available_bytes"] = available
            system["pswpout_start_pages"] = pswpout
        system["min_available_bytes"] = min(system["min_available_bytes"], available)
        system["pswpout_end_pages"] = pswpout
        system["pswpout_delta_pages"] = max(0, pswpout - system["pswpout_start_pages"])
        if process["state"] in ("Z", "X"):
            return "engine_zombie"
        if process.get("VmSwap", 0) > 0:
            return "engine_swapped"
        if process.get("VmRSS", 0) >= limits["rss_bytes"]:
            return "rss_limit"
        if available < limits["min_available_bytes"]:
            return "mem_available_low"
        return None

    try:
        reason = sample()
        write_log(log_path, record)
        os.write(ready_fd, b"R")
        os.close(ready_fd)
    except BaseException:
        try:
            os.write(ready_fd, b"E")
        except OSError:
            pass
        os._exit(2)

    while reason is None:
        time.sleep(limits["poll_ms"] / 1000)
        try:
            reason = sample()
        except (FileNotFoundError, ProcessLookupError):
            reason = "engine_exited"
        except BaseException as error:
            reason = "monitor_error"
            record["monitor_error"] = f"{type(error).__name__}: {error}"

    record["termination"]["reason"] = reason
    record["finished_unix"] = time.time()
    if reason in ("engine_exited", "engine_zombie", "pid_reused"):
        record["status"] = "stopped"
        write_log(log_path, record)
        os._exit(0)

    record["status"] = "triggered"
    record["termination"]["signal_sent"] = "SIGTERM"
    write_log(log_path, record)
    os.killpg(pgid, signal.SIGTERM)
    time.sleep(0.25)
    try:
        sample()
    except (FileNotFoundError, ProcessLookupError):
        pass
    record["finished_unix"] = time.time()
    record["termination"]["signal_sent"] = "SIGKILL"
    write_log(log_path, record)
    os.killpg(pgid, signal.SIGKILL)
    os._exit(3)


def main():
    binary = os.environ.get("BENCH_ENGINE_BINARY")
    log_name = os.environ.get("BENCH_MEMORY_GUARD_LOG")
    if not binary or not log_name:
        raise RuntimeError("BENCH_ENGINE_BINARY and BENCH_MEMORY_GUARD_LOG are required")
    limits = {"rss_bytes": 48 * GIB, "min_available_bytes": 8 * GIB, "poll_ms": 50}
    if os.environ.get("BENCH_MEMORY_GUARD_TEST") == "1":
        for key in limits:
            value = os.environ.get("BENCH_MEMORY_GUARD_TEST_" + key.upper())
            if value is not None:
                limits[key] = int(value)
        if min(limits.values()) < 0 or limits["poll_ms"] < 1 or limits["rss_bytes"] < 1:
            raise ValueError("invalid test guard threshold")
        limits["test_overrides"] = True
    log_path = Path(log_name).absolute()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.setsid()
    except PermissionError:
        if os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid():
            raise RuntimeError("cannot create a private engine session")
    pid = os.getpid()
    start_ticks = process_sample(pid)["start_ticks"]
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        guardian(pid, pid, start_ticks, log_path, limits, write_fd)
        os._exit(2)
    os.close(write_fd)
    ready = os.read(read_fd, 1)
    os.close(read_fd)
    if ready != b"R":
        os.waitpid(child, 0)
        raise RuntimeError("memory guardian failed to start")
    os.execvpe(binary, [binary, *sys.argv[1:]], os.environ)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ram_guard_exec: {error}", file=sys.stderr)
        sys.exit(2)
