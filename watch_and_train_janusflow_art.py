"""Watch local JanusFlow-Art weights and launch training once ready."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_JANUS_BYTES = 4_092_812_280
DEFAULT_VAE_BYTES = 334_641_164


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the watch-and-train launcher."""

    parser = argparse.ArgumentParser(description="Wait for local weights and launch JanusFlow-Art training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--expected-model-bytes", type=int, default=DEFAULT_JANUS_BYTES)
    parser.add_argument("--expected-vae-bytes", type=int, default=DEFAULT_VAE_BYTES)
    parser.add_argument("--log-path", required=True)
    return parser.parse_args()


def file_size(path: Path) -> int:
    """Return file size or zero when the file is absent."""

    return path.stat().st_size if path.exists() else 0


def main() -> None:
    """Wait until local weights are complete, then start training."""

    args = parse_args()
    model_file = Path(args.model_path) / "model.safetensors"
    vae_file = Path(args.vae_path) / "sdxl_vae.safetensors"
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log_handle:
        while True:
            model_bytes = file_size(model_file)
            vae_bytes = file_size(vae_file)
            status = {
                "event": "watch_progress",
                "model_bytes": model_bytes,
                "expected_model_bytes": int(args.expected_model_bytes),
                "vae_bytes": vae_bytes,
                "expected_vae_bytes": int(args.expected_vae_bytes),
            }
            print(json.dumps(status), flush=True)
            log_handle.write(json.dumps(status) + "\n")
            log_handle.flush()
            if model_bytes >= int(args.expected_model_bytes) and vae_bytes >= int(args.expected_vae_bytes):
                break
            time.sleep(max(int(args.poll_sec), 1))

        launch_record = {
            "event": "launch_training",
            "config": args.config,
            "model_path": args.model_path,
            "vae_path": args.vae_path,
            "output_dir": args.output_dir,
            "max_steps": int(args.max_steps),
        }
        print(json.dumps(launch_record), flush=True)
        log_handle.write(json.dumps(launch_record) + "\n")
        log_handle.flush()

        command = [
            "python",
            "train_janusflow_art.py",
            "--config",
            args.config,
            "--model-path",
            args.model_path,
            "--vae-path",
            args.vae_path,
            "--output-dir",
            args.output_dir,
            "--max-steps",
            str(int(args.max_steps)),
        ]
        subprocess.run(command, check=False, cwd="/root/autodl-tmp/repos/Janus")


if __name__ == "__main__":
    main()
