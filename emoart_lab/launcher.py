import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from emoart_lab.io_utils import load_json, write_json
from emoart_lab.layout import resolve_run_dir


def load_run_spec(track_dir: Path, expert_name: str, run_name: str) -> Dict[str, Any]:
    return load_json(resolve_run_dir(track_dir, expert_name, run_name) / "run.json")


def read_track_index(track_dir: Path) -> Dict[str, Any]:
    return load_json(track_dir / "track_index.json")


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_process_status(run_dir: Path) -> Dict[str, Any]:
    process_path = run_dir / "process.json"
    if not process_path.exists():
        return {}
    return load_json(process_path)


def launch_run(track_dir: Path, expert_name: str, run_name: str, background: bool) -> Dict[str, Any]:
    spec = load_run_spec(track_dir, expert_name, run_name)
    run_dir = resolve_run_dir(track_dir, expert_name, run_name)
    log_path = Path(spec["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing_status = read_process_status(run_dir)
    existing_pid = int(existing_status.get("pid", 0) or 0)
    if existing_pid and pid_is_running(existing_pid):
        raise RuntimeError(
            f"Run '{expert_name}/{run_name}' already has an active process: pid={existing_pid}"
        )

    if background:
        log_file = log_path.open("ab")
        process = subprocess.Popen(
            spec["command"],
            cwd=spec["cwd"],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            close_fds=True,
        )
        status = {
            "run_name": run_name,
            "expert_name": expert_name,
            "pid": process.pid,
            "log_path": str(log_path),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": spec["command"],
        }
        write_json(run_dir / "process.json", status)
        return status

    with log_path.open("ab") as log_file:
        completed = subprocess.run(
            spec["command"],
            cwd=spec["cwd"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    status = {
        "run_name": run_name,
        "expert_name": expert_name,
        "returncode": completed.returncode,
        "command": spec["command"],
        "log_path": str(log_path),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(run_dir / "process.json", status)
    return status
