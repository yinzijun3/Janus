import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from emoart_lab.download import download_model_snapshot
from emoart_lab.io_utils import load_json, write_json
from emoart_lab.launcher import launch_run, load_run_spec, pid_is_running, read_process_status
from emoart_lab.layout import resolve_run_dir
from emoart_lab.schemas import ContinuationConfig, ModelConfig, ProjectConfig, RunSpec, TrackConfig


Validator = Callable[[RunSpec], Dict[str, Any]]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def continuation_dir(track_dir: Path) -> Path:
    return track_dir / "system" / "continuation"


def continuation_log_path(track_dir: Path) -> Path:
    return continuation_dir(track_dir) / "continuation.log"


def continuation_state_path(track_dir: Path) -> Path:
    return continuation_dir(track_dir) / "state.json"


def continuation_process_path(track_dir: Path) -> Path:
    return continuation_dir(track_dir) / "process.json"


def run_spec_from_payload(payload: Dict[str, Any]) -> RunSpec:
    core_keys = {
        "run_name",
        "run_type",
        "expert_name",
        "stage_name",
        "cwd",
        "output_dir",
        "log_path",
        "command",
    }
    return RunSpec(
        run_name=payload["run_name"],
        run_type=payload["run_type"],
        expert_name=payload["expert_name"],
        stage_name=payload["stage_name"],
        cwd=payload["cwd"],
        output_dir=payload["output_dir"],
        log_path=payload["log_path"],
        command=list(payload["command"]),
        metadata={key: value for key, value in payload.items() if key not in core_keys},
    )


def load_prepared_configs(track_dir: Path) -> Tuple[ProjectConfig, TrackConfig]:
    config_dir = track_dir / "configs"
    project_config = ProjectConfig.from_dict(load_json(config_dir / "project_snapshot.json"))
    track_config = TrackConfig.from_dict(load_json(config_dir / "track_snapshot.json"))
    track_config.validate()
    return project_config, track_config


def read_state(track_dir: Path) -> Dict[str, Any]:
    path = continuation_state_path(track_dir)
    if not path.exists():
        return {"events": []}
    return load_json(path)


def record_event(track_dir: Path, phase: str, message: str, **extra: Any) -> None:
    state = read_state(track_dir)
    event = {
        "timestamp": utc_now(),
        "phase": phase,
        "message": message,
    }
    if extra:
        event["details"] = extra
    state.setdefault("events", []).append(event)
    state["last_phase"] = phase
    state["updated_at"] = event["timestamp"]
    write_json(continuation_state_path(track_dir), state)


def validate_model_snapshot(model_config: ModelConfig) -> Dict[str, Any]:
    model_path = Path(model_config.model_path)
    download_config = model_config.download
    required_files = list(download_config.required_files) if download_config is not None else []
    expected_sizes = dict(download_config.expected_sizes) if download_config is not None else {}

    missing_files: List[str] = []
    size_mismatches: Dict[str, Dict[str, int]] = {}
    actual_sizes: Dict[str, int] = {}
    for filename in sorted(set(required_files) | set(expected_sizes.keys())):
        path = model_path / filename
        if not path.exists():
            missing_files.append(filename)
            continue
        size = path.stat().st_size
        actual_sizes[filename] = size
        expected_size = expected_sizes.get(filename)
        if expected_size is not None and size != expected_size:
            size_mismatches[filename] = {
                "expected": expected_size,
                "actual": size,
            }

    return {
        "model_path": str(model_path),
        "required_files": required_files,
        "expected_sizes": expected_sizes,
        "actual_sizes": actual_sizes,
        "missing_files": missing_files,
        "size_mismatches": size_mismatches,
        "is_complete": model_path.exists() and not missing_files and not size_mismatches,
        "checked_at": utc_now(),
    }


def ensure_local_model_snapshot(track_dir: Path, track_config: TrackConfig) -> Dict[str, Any]:
    status = validate_model_snapshot(track_config.model)
    write_json(continuation_dir(track_dir) / "model_snapshot_status.json", status)
    if status["is_complete"]:
        record_event(track_dir, "model_ready", "Local 7B snapshot already complete.", status_path="model_snapshot_status.json")
        return status

    download_config = track_config.model.download
    if download_config is None:
        raise RuntimeError(
            f"Model path {track_config.model.model_path} is incomplete and no download config is defined."
        )

    record_event(
        track_dir,
        "download",
        "Local 7B snapshot incomplete; resuming mirror download.",
        model_path=track_config.model.model_path,
    )
    summary = download_model_snapshot(
        endpoint=download_config.endpoint,
        repo_id=download_config.repo_id,
        output_dir=Path(track_config.model.model_path),
        revision=download_config.revision,
        weight_parallelism=download_config.weight_parallelism,
        max_retries=download_config.max_retries,
        trust_env_proxy=not download_config.ignore_env_proxy,
    )
    write_json(continuation_dir(track_dir) / "download_resume_summary.json", summary)
    status = validate_model_snapshot(track_config.model)
    write_json(continuation_dir(track_dir) / "model_snapshot_status.json", status)
    if not status["is_complete"]:
        raise RuntimeError("Model download finished but the local snapshot still failed validation.")
    record_event(track_dir, "model_ready", "Local 7B snapshot passed exact-byte validation.")
    return status


def run_model_smoke_check(track_dir: Path, project_config: ProjectConfig, track_config: TrackConfig) -> Dict[str, Any]:
    smoke_dir = continuation_dir(track_dir) / "model_smoke"
    summary_path = smoke_dir / "summary.json"
    log_path = smoke_dir / "launch.log"
    if summary_path.exists():
        summary = load_json(summary_path)
        if summary.get("status") == "ok" and summary.get("model_path") == track_config.model.model_path:
            record_event(track_dir, "smoke_check", "Local 7B smoke check already passed.", summary_path=str(summary_path))
            return summary

    smoke_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "emoart_lab.smoke",
        "--model-path",
        track_config.model.model_path,
        "--output-json",
        str(summary_path),
    ]
    record_event(track_dir, "smoke_check", "Running local-only VLChatProcessor / AutoModel smoke check.")
    with log_path.open("ab") as log_file:
        completed = subprocess.run(
            command,
            cwd=project_config.repo_dir,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Model smoke check failed with returncode={completed.returncode}. See {log_path}")
    summary = load_json(summary_path)
    if summary.get("status") != "ok":
        raise RuntimeError(f"Model smoke check did not report success. See {summary_path}")
    record_event(track_dir, "smoke_check", "Local 7B smoke check passed.", summary_path=str(summary_path))
    return summary


def extract_run_model_path(run_spec: Dict[str, Any]) -> str:
    command = list(run_spec.get("command", []))
    for idx, token in enumerate(command[:-1]):
        if token == "--model-path":
            return str(command[idx + 1])
    return ""


def validate_materialized_model_paths(track_dir: Path, model_path: str) -> None:
    mismatches: List[Dict[str, str]] = []
    runs_dir = track_dir / "runs"
    for run_json in sorted(runs_dir.glob("*/*/run.json")):
        payload = load_json(run_json)
        run_model_path = extract_run_model_path(payload)
        if run_model_path and run_model_path != model_path:
            mismatches.append(
                {
                    "run_json": str(run_json),
                    "model_path": run_model_path,
                }
            )
    if mismatches:
        raise RuntimeError(
            "Materialized run specs still point to a stale model path. Re-run prepare first. "
            + json.dumps(mismatches[:5], ensure_ascii=False)
        )


def compare_run_name(stage_name: str, manifest_source: str, num_samples: int) -> str:
    return f"compare_{stage_name}_{manifest_source}_{num_samples}"


def packet_run_name(compare_name: str) -> str:
    return f"{compare_name}_packet"


def parse_train_log(path: Path) -> Dict[str, Any]:
    nonempty_lines = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        nonempty_lines += 1
        payload = json.loads(text)
        if contains_nonfinite(payload):
            return {
                "ok": False,
                "reason": "non_finite_value",
                "line_payload": payload,
            }
    return {
        "ok": nonempty_lines > 0,
        "reason": "" if nonempty_lines > 0 else "empty_log",
    }


def contains_nonfinite(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return False
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_nonfinite(item) for item in value)
    return False


def validate_train_run(run_spec: RunSpec) -> Dict[str, Any]:
    output_dir = Path(run_spec.output_dir)
    adapter_dir = output_dir / "final_adapter"
    log_path = output_dir / "train_log.jsonl"
    config_path = output_dir / "train_config.json"
    missing: List[str] = []
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        missing.append(str(adapter_dir))
    if not log_path.exists():
        missing.append(str(log_path))
    if not config_path.exists():
        missing.append(str(config_path))

    log_check = {"ok": False, "reason": "missing"}
    if log_path.exists():
        log_check = parse_train_log(log_path)

    return {
        "run_name": run_spec.run_name,
        "run_type": run_spec.run_type,
        "output_dir": run_spec.output_dir,
        "missing": missing,
        "train_log_check": log_check,
        "is_complete": not missing and log_check.get("ok", False),
    }


def validate_compare_run(run_spec: RunSpec) -> Dict[str, Any]:
    output_dir = Path(run_spec.output_dir)
    summary_path = output_dir / "summary.json"
    comparison_path = output_dir / "comparison.jsonl"
    missing = [str(path) for path in [summary_path, comparison_path] if not path.exists()]
    return {
        "run_name": run_spec.run_name,
        "run_type": run_spec.run_type,
        "output_dir": run_spec.output_dir,
        "missing": missing,
        "is_complete": not missing,
    }


def validate_packet_run(run_spec: RunSpec) -> Dict[str, Any]:
    output_dir = Path(run_spec.output_dir)
    review_path = output_dir / "review_packet.md"
    manual_path = output_dir / "manual_review_sheet.md"
    missing = [str(path) for path in [review_path, manual_path] if not path.exists()]
    return {
        "run_name": run_spec.run_name,
        "run_type": run_spec.run_type,
        "output_dir": run_spec.output_dir,
        "missing": missing,
        "is_complete": not missing,
    }


def archive_incomplete_run(run_dir: Path) -> Path:
    retry_root = run_dir / "retries"
    archive_dir = retry_root / utc_now().replace(":", "").replace("-", "")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for name in ["artifacts", "launch.log", "process.json"]:
        path = run_dir / name
        if path.exists():
            shutil.move(str(path), str(archive_dir / name))
    return archive_dir


def execute_run(
    track_dir: Path,
    expert_name: str,
    run_name: str,
    validator: Validator,
    label: str,
) -> Dict[str, Any]:
    run_spec = run_spec_from_payload(load_run_spec(track_dir, expert_name, run_name))
    run_dir = resolve_run_dir(track_dir, expert_name, run_name)
    validation = validator(run_spec)
    if validation["is_complete"]:
        record_event(track_dir, "run_skip", f"{label} already complete.", expert=expert_name, run_name=run_name)
        return {
            "status": "skipped",
            "validation": validation,
        }

    process_status = read_process_status(run_dir)
    existing_pid = int(process_status.get("pid", 0) or 0)
    if existing_pid and pid_is_running(existing_pid):
        raise RuntimeError(f"{label} already has an active pid={existing_pid}; refusing to start a duplicate.")

    archive_dir = archive_incomplete_run(run_dir)
    record_event(
        track_dir,
        "run_start",
        f"Launching {label}.",
        expert=expert_name,
        run_name=run_name,
        archived_to=str(archive_dir),
    )
    status = launch_run(track_dir=track_dir, expert_name=expert_name, run_name=run_name, background=False)
    if status.get("returncode") != 0:
        raise RuntimeError(f"{label} failed with returncode={status.get('returncode')}. See {run_spec.log_path}")

    validation = validator(run_spec)
    if not validation["is_complete"]:
        raise RuntimeError(f"{label} finished but output validation failed: {json.dumps(validation, ensure_ascii=False)}")

    record_event(track_dir, "run_done", f"{label} completed.", expert=expert_name, run_name=run_name)
    return {
        "status": "completed",
        "validation": validation,
    }


def run_stage1_smoke_sequence(track_dir: Path, continuation: ContinuationConfig, expert_name: str) -> None:
    train_run_name = f"train_{continuation.train_stage}"
    execute_run(
        track_dir=track_dir,
        expert_name=expert_name,
        run_name=train_run_name,
        validator=validate_train_run,
        label=f"{expert_name} / {train_run_name}",
    )
    for evaluation in continuation.smoke_evaluations:
        compare_name = compare_run_name(continuation.train_stage, evaluation.manifest_source, evaluation.num_samples)
        execute_run(
            track_dir=track_dir,
            expert_name=expert_name,
            run_name=compare_name,
            validator=validate_compare_run,
            label=f"{expert_name} / {compare_name}",
        )
        if evaluation.build_packet:
            execute_run(
                track_dir=track_dir,
                expert_name=expert_name,
                run_name=packet_run_name(compare_name),
                validator=validate_packet_run,
                label=f"{expert_name} / {packet_run_name(compare_name)}",
            )


def run_post_stage_evaluations(track_dir: Path, continuation: ContinuationConfig, expert_name: str) -> None:
    for evaluation in continuation.post_stage_evaluations:
        compare_name = compare_run_name(continuation.train_stage, evaluation.manifest_source, evaluation.num_samples)
        execute_run(
            track_dir=track_dir,
            expert_name=expert_name,
            run_name=compare_name,
            validator=validate_compare_run,
            label=f"{expert_name} / {compare_name}",
        )
        if evaluation.build_packet:
            execute_run(
                track_dir=track_dir,
                expert_name=expert_name,
                run_name=packet_run_name(compare_name),
                validator=validate_packet_run,
                label=f"{expert_name} / {packet_run_name(compare_name)}",
            )


def continue_prepared_track(track_dir: Path) -> Dict[str, Any]:
    track_dir = track_dir.resolve()
    continuation_dir(track_dir).mkdir(parents=True, exist_ok=True)
    project_config, track_config = load_prepared_configs(track_dir)
    if track_config.continuation is None:
        raise RuntimeError(f"Track {track_config.track_name} has no continuation config.")
    if track_config.continuation.auto_advance_to_stage2:
        raise NotImplementedError("auto_advance_to_stage2 is not implemented in the continuation supervisor.")

    record_event(track_dir, "start", "Starting prepared-track continuation supervisor.")
    validate_materialized_model_paths(track_dir, track_config.model.model_path)
    model_status = ensure_local_model_snapshot(track_dir, track_config)
    smoke_summary = run_model_smoke_check(track_dir, project_config, track_config)

    for expert_name in track_config.continuation.expert_order:
        run_stage1_smoke_sequence(track_dir, track_config.continuation, expert_name)

    for expert_name in track_config.continuation.expert_order:
        run_post_stage_evaluations(track_dir, track_config.continuation, expert_name)

    result = {
        "status": "completed",
        "track_dir": str(track_dir),
        "model_snapshot_status": model_status,
        "smoke_summary_path": str(continuation_dir(track_dir) / "model_smoke" / "summary.json"),
        "completed_at": utc_now(),
    }
    write_json(continuation_dir(track_dir) / "result.json", result)
    record_event(track_dir, "complete", "Continuation queue finished without entering stage2.")
    return result


def launch_continuation_supervisor(track_dir: Path) -> Dict[str, Any]:
    track_dir = track_dir.resolve()
    continuation_dir(track_dir).mkdir(parents=True, exist_ok=True)
    existing = {}
    process_path = continuation_process_path(track_dir)
    if process_path.exists():
        existing = load_json(process_path)
    existing_pid = int(existing.get("pid", 0) or 0)
    if existing_pid and pid_is_running(existing_pid):
        raise RuntimeError(f"Continuation supervisor is already running with pid={existing_pid}.")

    project_config, _ = load_prepared_configs(track_dir)
    command = [
        sys.executable,
        "-m",
        "emoart_lab.cli",
        "continue-track",
        "--track-dir",
        str(track_dir),
        "--worker",
    ]
    log_path = continuation_log_path(track_dir)
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=project_config.repo_dir,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            close_fds=True,
        )
    status = {
        "pid": process.pid,
        "track_dir": str(track_dir),
        "log_path": str(log_path),
        "command": command,
        "started_at": utc_now(),
    }
    write_json(process_path, status)
    record_event(track_dir, "supervisor_start", "Detached continuation supervisor launched.", pid=process.pid)
    return status
