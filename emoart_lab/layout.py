from pathlib import Path

from emoart_lab.schemas import ProjectConfig, TrackConfig


def resolve_track_dir(project_config: ProjectConfig, track_config: TrackConfig) -> Path:
    return Path(project_config.output_root) / project_config.project_name / track_config.track_name


def resolve_family_manifest_path(track_dir: Path, expert_name: str, split_name: str) -> Path:
    return track_dir / "data" / "family_manifests" / expert_name / f"{split_name}.jsonl"


def resolve_texture_manifest_path(track_dir: Path, expert_name: str) -> Path:
    return track_dir / "data" / "texture_refine" / expert_name / "train.jsonl"


def resolve_run_dir(track_dir: Path, expert_name: str, run_name: str) -> Path:
    return track_dir / "runs" / expert_name / run_name


def resolve_stage_artifact_dir(track_dir: Path, expert_name: str, stage_name: str) -> Path:
    return resolve_run_dir(track_dir, expert_name, f"train_{stage_name}") / "artifacts"


def resolve_stage_adapter_dir(track_dir: Path, expert_name: str, stage_name: str) -> Path:
    return resolve_stage_artifact_dir(track_dir, expert_name, stage_name) / "final_adapter"

