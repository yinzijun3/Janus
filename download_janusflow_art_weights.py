"""Download JanusFlow-Art base weights with resumable direct transfers.

This helper avoids repeated manual retry loops when Hub front-door requests are
unstable. It resolves official Hugging Face `resolve/main/...` URLs, then streams
the redirected object-store URLs with resumable range requests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urljoin

import requests


JANUSFLOW_FILES = {
    "config.json": "https://huggingface.co/deepseek-ai/JanusFlow-1.3B/resolve/main/config.json",
    "model.safetensors": "https://huggingface.co/deepseek-ai/JanusFlow-1.3B/resolve/main/model.safetensors",
    "processor_config.json": "https://huggingface.co/deepseek-ai/JanusFlow-1.3B/resolve/main/processor_config.json",
    "preprocessor_config.json": "https://huggingface.co/deepseek-ai/JanusFlow-1.3B/resolve/main/preprocessor_config.json",
    "special_tokens_map.json": "https://huggingface.co/deepseek-ai/JanusFlow-1.3B/resolve/main/special_tokens_map.json",
    "tokenizer.json": "https://huggingface.co/deepseek-ai/JanusFlow-1.3B/resolve/main/tokenizer.json",
}

SDXL_VAE_FILES = {
    "config.json": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/config.json",
    "sdxl_vae.safetensors": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
}


def parse_expected_total_bytes(total_header: str | None, *, status_code: int, existing_bytes: int) -> int | None:
    """Infer the expected final file size from HTTP response metadata."""

    if not total_header:
        return None
    if total_header.startswith("bytes "):
        total_value = total_header.split("/")[-1]
        return int(total_value) if total_value.isdigit() else None
    if total_header.isdigit():
        content_length = int(total_header)
        if status_code == 206:
            return existing_bytes + content_length
        return content_length
    return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for weight download targets."""

    parser = argparse.ArgumentParser(description="Download JanusFlow-Art weights.")
    parser.add_argument(
        "--target",
        choices=["all", "janusflow", "vae"],
        default="all",
        help="Which weight bundle to download.",
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/model_cache/janusflow_art",
        help="Directory where local model folders will be created.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=200,
        help="Maximum number of retry attempts for streamed downloads.",
    )
    parser.add_argument(
        "--report-interval-sec",
        type=float,
        default=5.0,
        help="Seconds between progress prints during long downloads.",
    )
    return parser.parse_args()


def stream_download(
    session: requests.Session,
    *,
    resolve_url: str,
    output_path: Path,
    max_retries: int,
    report_interval_sec: float,
) -> None:
    """Download one file with resume support and periodic progress logging."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 1024 * 1024

    for attempt in range(1, max_retries + 1):
        downloaded = output_path.stat().st_size if output_path.exists() else 0
        try:
            metadata_response = session.get(resolve_url, allow_redirects=False, timeout=30)
            metadata_response.raise_for_status()
            direct_url = metadata_response.headers.get("location", resolve_url)
            direct_url = urljoin(resolve_url, direct_url)

            headers: Dict[str, str] = {}
            file_mode = "wb"
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                file_mode = "ab"

            with session.get(
                direct_url,
                stream=True,
                timeout=(30, 120),
                headers=headers,
            ) as response:
                if response.status_code == 416 and downloaded > 0:
                    print(
                        json.dumps(
                            {
                                "event": "download_complete",
                                "path": str(output_path),
                                "bytes": downloaded,
                                "note": "server reported requested range already satisfied",
                            }
                        ),
                        flush=True,
                    )
                    return
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"unexpected status {response.status_code}")

                total_header = response.headers.get("Content-Range") or response.headers.get("Content-Length")
                expected_total_bytes = parse_expected_total_bytes(
                    total_header,
                    status_code=response.status_code,
                    existing_bytes=downloaded,
                )
                if response.status_code == 200 and downloaded > 0:
                    # Server ignored the range request; restart this file cleanly.
                    downloaded = 0
                    file_mode = "wb"
                print(
                    json.dumps(
                        {
                            "event": "download_open",
                            "path": str(output_path),
                            "attempt": attempt,
                            "status": response.status_code,
                            "existing_bytes": downloaded,
                            "reported_total": total_header,
                            "expected_total_bytes": expected_total_bytes,
                        }
                    ),
                    flush=True,
                )
                last_report = time.time()
                with open(output_path, file_mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_report >= report_interval_sec:
                            print(
                                json.dumps(
                                    {
                                        "event": "download_progress",
                                        "path": str(output_path),
                                        "attempt": attempt,
                                        "downloaded_bytes": downloaded,
                                    }
                                ),
                                flush=True,
                            )
                            last_report = now
            final_size = output_path.stat().st_size
            if expected_total_bytes is not None and final_size != expected_total_bytes:
                raise RuntimeError(
                    f"incomplete download: expected {expected_total_bytes} bytes, got {final_size}"
                )
            print(
                json.dumps(
                    {
                        "event": "download_complete",
                        "path": str(output_path),
                        "bytes": final_size,
                    }
                ),
                flush=True,
            )
            return
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "download_retry",
                        "path": str(output_path),
                        "attempt": attempt,
                        "existing_bytes": downloaded,
                        "error": str(exc),
                    }
                ),
                flush=True,
            )
            time.sleep(5)

    raise RuntimeError(f"failed to download {output_path} after {max_retries} attempts")


def iter_targets(target: str) -> Iterable[tuple[str, Dict[str, str]]]:
    """Yield target bundles selected by the CLI."""

    if target in {"all", "janusflow"}:
        yield "JanusFlow-1.3B", JANUSFLOW_FILES
    if target in {"all", "vae"}:
        yield "sdxl-vae", SDXL_VAE_FILES


def main() -> None:
    """Download the requested JanusFlow-Art weight bundles."""

    args = parse_args()
    output_root = Path(args.output_root)
    session = requests.Session()

    for directory_name, files in iter_targets(args.target):
        bundle_root = output_root / directory_name
        bundle_root.mkdir(parents=True, exist_ok=True)
        for filename, resolve_url in files.items():
            stream_download(
                session,
                resolve_url=resolve_url,
                output_path=bundle_root / filename,
                max_retries=args.max_retries,
                report_interval_sec=args.report_interval_sec,
            )


if __name__ == "__main__":
    main()
