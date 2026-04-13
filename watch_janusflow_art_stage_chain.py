"""Watch one JanusFlow-Art stage, then evaluate it and launch the next stage.

This helper keeps the handoff logic outside the core trainer so staged research
runs can chain together without modifying the existing training entrypoints.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the stage-chain watcher."""

    parser = argparse.ArgumentParser(description="Watch a JanusFlow-Art stage and continue to the next one.")
    parser.add_argument("--wait-output-dir", required=True, help="Stage output directory to watch for final_checkpoint.")
    parser.add_argument("--eval-config", required=True, help="Config used for the evaluation run.")
    parser.add_argument("--next-config", required=True, help="Config used for the next training stage.")
    parser.add_argument("--next-output-dir", required=True, help="Output directory for the next training stage.")
    parser.add_argument("--init-output-dir", required=True, help="Clean checkpoint directory passed to the next stage.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


def log(message: str) -> None:
    """Print one timestamped log line."""

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{timestamp}] {message}", flush=True)


def wait_for_final_checkpoint(output_dir: Path, poll_seconds: float) -> Path:
    """Poll until one stage writes its final checkpoint summary."""

    checkpoint_dir = output_dir / "final_checkpoint"
    summary_path = checkpoint_dir / "checkpoint_summary.json"
    while True:
        if summary_path.exists():
            log(f"found_final_checkpoint {checkpoint_dir}")
            return checkpoint_dir
        log(f"waiting_for_final_checkpoint {output_dir}")
        time.sleep(poll_seconds)


def copy_checkpoint_without_optimizer(source_dir: Path, target_dir: Path, skip_names: Iterable[str]) -> None:
    """Copy one checkpoint tree while dropping optimizer state files."""

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    skip = set(skip_names)
    for source_path in source_dir.rglob("*"):
        relative = source_path.relative_to(source_dir)
        if any(part in skip for part in relative.parts):
            continue
        target_path = target_dir / relative
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    log(f"prepared_init_checkpoint {target_dir}")


def run_command(command: list[str]) -> None:
    """Run one subprocess and raise on failure."""

    log("run_command " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    """Wait for the current stage, evaluate it, then launch the next one."""

    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    wait_output_dir = Path(args.wait_output_dir)
    final_checkpoint = wait_for_final_checkpoint(wait_output_dir, args.poll_seconds)

    run_command(
        [
            sys.executable,
            str(repo_root / "eval_janusflow_art.py"),
            "--config",
            args.eval_config,
            "--checkpoint",
            str(final_checkpoint),
            "--output-dir",
            str(wait_output_dir),
            "--model-path",
            args.model_path,
            "--vae-path",
            args.vae_path,
        ]
    )

    init_output_dir = Path(args.init_output_dir)
    copy_checkpoint_without_optimizer(
        final_checkpoint,
        init_output_dir,
        skip_names=("optimizer.pt", "scheduler.pt"),
    )

    run_command(
        [
            sys.executable,
            str(repo_root / "train_janusflow_art.py"),
            "--config",
            args.next_config,
            "--resume-from-checkpoint",
            str(init_output_dir),
            "--output-dir",
            args.next_output_dir,
            "--model-path",
            args.model_path,
            "--vae-path",
            args.vae_path,
        ]
    )


if __name__ == "__main__":
    main()
