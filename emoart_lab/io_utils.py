import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_cli_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def cli_args_from_mapping(values: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for key, value in values.items():
        if value is None:
            continue
        flag = normalize_cli_flag(key)
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            continue
        if isinstance(value, list):
            if not value:
                continue
            parts.append(flag)
            parts.extend(str(item) for item in value)
            continue
        parts.extend([flag, str(value)])
    return parts


def render_shell_command(command: List[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in command)


def resolve_runtime_python() -> str:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return sys.executable
