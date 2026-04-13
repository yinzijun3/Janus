import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from emoart_lab.io_utils import write_json


DEFAULT_REQUIRED_FILES = [
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def api_model_url(endpoint: str, repo_id: str) -> str:
    return endpoint.rstrip("/") + f"/api/models/{repo_id}"


def resolve_file_url(endpoint: str, repo_id: str, filename: str, revision: str = "main") -> str:
    return endpoint.rstrip("/") + f"/{repo_id}/resolve/{revision}/{filename}"


def list_repo_files(endpoint: str, repo_id: str, timeout: int = 30) -> Dict[str, object]:
    response = requests.get(api_model_url(endpoint, repo_id), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    siblings = [item["rfilename"] for item in payload.get("siblings", [])]
    return {
        "repo_id": repo_id,
        "sha": payload.get("sha"),
        "siblings": siblings,
        "raw": payload,
    }


def required_model_files(repo_files: Iterable[str]) -> List[str]:
    files = list(repo_files)
    required = list(DEFAULT_REQUIRED_FILES)
    weight_files = sorted([name for name in files if name.startswith("pytorch_model-") and name.endswith(".bin")])
    required.extend(weight_files)
    return required


def split_model_files(filenames: Iterable[str]) -> Tuple[List[str], List[str]]:
    small_files: List[str] = []
    weight_files: List[str] = []
    for filename in filenames:
        if filename.startswith("pytorch_model-") and filename.endswith(".bin"):
            weight_files.append(filename)
        else:
            small_files.append(filename)
    return small_files, sorted(weight_files)


def stream_download(
    url: str,
    output_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
    timeout: int = 30,
    max_retries: int = 12,
    retry_backoff_sec: float = 5.0,
    progress_interval_sec: float = 10.0,
    trust_env_proxy: bool = True,
) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = trust_env_proxy
    total_size: Optional[int] = None
    total_bytes_written = 0
    started_at = time.time()
    last_report = started_at
    attempts = 0
    last_url = url

    while True:
        existing_size = output_path.stat().st_size if output_path.exists() else 0
        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(10, timeout),
                allow_redirects=True,
            ) as response:
                last_url = response.url
                if response.status_code == 416:
                    final_size = output_path.stat().st_size if output_path.exists() else 0
                    return {
                        "path": str(output_path),
                        "status": "already_complete",
                        "bytes_written": 0,
                        "final_size": final_size,
                        "total_size": total_size,
                        "url": last_url,
                        "attempts": attempts,
                    }

                response.raise_for_status()
                if existing_size > 0 and response.status_code != 206:
                    existing_size = 0

                content_range = response.headers.get("Content-Range")
                content_length = response.headers.get("Content-Length")
                if content_range and "/" in content_range:
                    total_size = int(content_range.rsplit("/", 1)[-1])
                elif content_length:
                    current_length = int(content_length)
                    total_size = existing_size + current_length if response.status_code == 206 else current_length

                mode = "ab" if existing_size > 0 and response.status_code == 206 else "wb"
                with output_path.open(mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        total_bytes_written += len(chunk)
                        now = time.time()
                        if now - last_report >= progress_interval_sec:
                            current_size = output_path.stat().st_size if output_path.exists() else 0
                            rate_mib = (total_bytes_written / max(now - started_at, 1e-6)) / (1024 * 1024)
                            print(
                                {
                                    "file": output_path.name,
                                    "current_size_bytes": current_size,
                                    "total_size_bytes": total_size,
                                    "downloaded_this_run_bytes": total_bytes_written,
                                    "avg_rate_mib_s": round(rate_mib, 3),
                                    "attempts": attempts,
                                    "url": last_url,
                                },
                                flush=True,
                            )
                            last_report = now

                final_size = output_path.stat().st_size if output_path.exists() else 0
                if total_size is None or final_size >= total_size:
                    return {
                        "path": str(output_path),
                        "status": "downloaded",
                        "bytes_written": total_bytes_written,
                        "final_size": final_size,
                        "total_size": total_size,
                        "url": last_url,
                        "attempts": attempts,
                    }
        except requests.RequestException as exc:
            attempts += 1
            if attempts > max_retries:
                raise RuntimeError(f"Failed downloading {output_path.name} after {max_retries} retries: {exc}") from exc
            print(
                {
                    "file": output_path.name,
                    "status": "retrying",
                    "attempt": attempts,
                    "error": repr(exc),
                    "sleep_sec": retry_backoff_sec,
                },
                flush=True,
            )
            time.sleep(retry_backoff_sec)


def download_one_file(
    endpoint: str,
    repo_id: str,
    revision: str,
    output_dir: Path,
    filename: str,
    timeout: int,
    max_retries: int,
    trust_env_proxy: bool,
) -> Dict[str, object]:
    return stream_download(
        url=resolve_file_url(endpoint=endpoint, repo_id=repo_id, filename=filename, revision=revision),
        output_path=output_dir / filename,
        timeout=timeout,
        max_retries=max_retries,
        trust_env_proxy=trust_env_proxy,
    )


def download_many_files(
    endpoint: str,
    repo_id: str,
    revision: str,
    output_dir: Path,
    filenames: List[str],
    timeout: int,
    max_retries: int,
    parallel_workers: int,
    trust_env_proxy: bool,
) -> List[Dict[str, object]]:
    if not filenames:
        return []
    if parallel_workers <= 1 or len(filenames) == 1:
        return [
            download_one_file(
                endpoint=endpoint,
                repo_id=repo_id,
                revision=revision,
                output_dir=output_dir,
                filename=filename,
                timeout=timeout,
                max_retries=max_retries,
                trust_env_proxy=trust_env_proxy,
            )
            for filename in filenames
        ]

    results_by_name: Dict[str, Dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_map = {
            executor.submit(
                download_one_file,
                endpoint,
                repo_id,
                revision,
                output_dir,
                filename,
                timeout,
                max_retries,
                trust_env_proxy,
            ): filename
            for filename in filenames
        }
        for future in as_completed(future_map):
            filename = future_map[future]
            results_by_name[filename] = future.result()
    return [results_by_name[filename] for filename in filenames]


def write_progress_manifest(
    output_dir: Path,
    repo_id: str,
    endpoint: str,
    revision: str,
    filenames: List[str],
    results: List[Dict[str, object]],
) -> None:
    return {
        "repo_id": repo_id,
        "endpoint": endpoint,
        "revision": revision,
        "output_dir": str(output_dir),
        "filenames": filenames,
        "results": results,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def download_model_snapshot(
    endpoint: str,
    repo_id: str,
    output_dir: Path,
    revision: str = "main",
    timeout: int = 30,
    include_readme: bool = False,
    weight_parallelism: int = 2,
    max_retries: int = 12,
    trust_env_proxy: bool = True,
) -> Dict[str, object]:
    if trust_env_proxy:
        repo_info = list_repo_files(endpoint=endpoint, repo_id=repo_id, timeout=timeout)
    else:
        session = requests.Session()
        session.trust_env = False
        response = session.get(api_model_url(endpoint, repo_id), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        repo_info = {
            "repo_id": repo_id,
            "sha": payload.get("sha"),
            "siblings": [item["rfilename"] for item in payload.get("siblings", [])],
            "raw": payload,
        }
    filenames = required_model_files(repo_info["siblings"])
    if include_readme and "README.md" in repo_info["siblings"]:
        filenames.append("README.md")

    output_dir.mkdir(parents=True, exist_ok=True)
    small_files, weight_files = split_model_files(filenames)
    results: List[Dict[str, object]] = []
    results.extend(
        download_many_files(
            endpoint=endpoint,
            repo_id=repo_id,
            revision=revision,
            output_dir=output_dir,
            filenames=small_files,
            timeout=timeout,
            max_retries=max_retries,
            parallel_workers=1,
            trust_env_proxy=trust_env_proxy,
        )
    )
    write_json(
        output_dir / "download_progress.json",
        write_progress_manifest(
            output_dir=output_dir,
            repo_id=repo_id,
            endpoint=endpoint,
            revision=revision,
            filenames=filenames,
            results=results,
        ),
    )
    results.extend(
        download_many_files(
            endpoint=endpoint,
            repo_id=repo_id,
            revision=revision,
            output_dir=output_dir,
            filenames=weight_files,
            timeout=timeout,
            max_retries=max_retries,
            parallel_workers=weight_parallelism,
            trust_env_proxy=trust_env_proxy,
        )
    )

    summary = {
        "repo_id": repo_id,
        "endpoint": endpoint,
        "revision": revision,
        "output_dir": str(output_dir),
        "repo_sha": repo_info.get("sha"),
        "filenames": filenames,
        "results": results,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output_dir / "download_summary.json", summary)
    return summary
