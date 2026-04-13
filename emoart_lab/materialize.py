import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from emoart_lab.io_utils import (
    cli_args_from_mapping,
    load_json,
    render_shell_command,
    resolve_runtime_python,
    write_json,
)
from emoart_lab.layout import (
    resolve_family_manifest_path,
    resolve_run_dir,
    resolve_stage_adapter_dir,
    resolve_stage_artifact_dir,
    resolve_texture_manifest_path,
    resolve_track_dir,
)
from emoart_lab.manifests import build_family_manifests, build_texture_refine_manifests
from emoart_lab.schemas import ProjectConfig, RunSpec, TrackConfig, TrainStageConfig


def resolve_manifest_source(
    project_config: ProjectConfig,
    track_dir: Path,
    expert_name: str,
    source_name: str,
) -> Path:
    if source_name == "family_train":
        return resolve_family_manifest_path(track_dir, expert_name, "train")
    if source_name == "family_val":
        return resolve_family_manifest_path(track_dir, expert_name, "val")
    if source_name == "texture_train":
        return resolve_texture_manifest_path(track_dir, expert_name)
    if source_name == "full_train":
        return Path(project_config.data["base_train_manifest"])
    if source_name == "full_val":
        return Path(project_config.data["base_val_manifest"])
    raise ValueError(f"Unsupported manifest source: {source_name}")


def write_launch_script(path: Path, command: List[str], cwd: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {cwd}",
                render_shell_command(command),
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def build_train_run_spec(
    project_config: ProjectConfig,
    track_config: TrackConfig,
    track_dir: Path,
    expert_name: str,
    stage_config: TrainStageConfig,
) -> RunSpec:
    repo_dir = Path(project_config.repo_dir)
    run_name = f"train_{stage_config.name}"
    run_dir = resolve_run_dir(track_dir, expert_name, run_name)
    output_dir = resolve_stage_artifact_dir(track_dir, expert_name, stage_config.name)
    train_manifest = resolve_manifest_source(project_config, track_dir, expert_name, stage_config.train_manifest)
    val_manifest = resolve_manifest_source(project_config, track_dir, expert_name, stage_config.val_manifest)

    train_args = dict(stage_config.train_args)
    train_args.setdefault("dtype", track_config.model.dtype)
    runtime_python = resolve_runtime_python()
    command = [
        runtime_python,
        str(repo_dir / "train_emoart_gen_lora.py"),
        "--model-path",
        track_config.model.model_path,
        "--train-data",
        str(train_manifest),
        "--val-data",
        str(val_manifest),
        "--output-dir",
        str(output_dir),
    ]
    if stage_config.resume_from_stage:
        command.extend(
            [
                "--resume-from-checkpoint",
                str(resolve_stage_adapter_dir(track_dir, expert_name, stage_config.resume_from_stage)),
            ]
        )
    command.extend(cli_args_from_mapping(train_args))

    spec = RunSpec(
        run_name=run_name,
        run_type="train",
        expert_name=expert_name,
        stage_name=stage_config.name,
        cwd=str(repo_dir),
        output_dir=str(output_dir),
        log_path=str(run_dir / "launch.log"),
        command=command,
        metadata={
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "resume_from_stage": stage_config.resume_from_stage,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run.json", spec.to_dict())
    write_launch_script(run_dir / "launch.sh", command, repo_dir)
    return spec


def build_compare_run_spec(
    project_config: ProjectConfig,
    track_config: TrackConfig,
    track_dir: Path,
    expert_name: str,
    stage_name: str,
    manifest_source: str,
    num_samples: int,
) -> RunSpec:
    repo_dir = Path(project_config.repo_dir)
    run_name = f"compare_{stage_name}_{manifest_source}_{num_samples}"
    run_dir = resolve_run_dir(track_dir, expert_name, run_name)
    output_dir = run_dir / "artifacts"
    manifest_path = resolve_manifest_source(project_config, track_dir, expert_name, manifest_source)
    eval_args = dict(track_config.evaluation_defaults)
    eval_args.setdefault("dtype", track_config.model.dtype)
    runtime_python = resolve_runtime_python()
    command = [
        runtime_python,
        str(repo_dir / "compare_emoart_gen.py"),
        "--model-path",
        track_config.model.model_path,
        "--manifest-path",
        str(manifest_path),
        "--adapter-path",
        str(resolve_stage_adapter_dir(track_dir, expert_name, stage_name)),
        "--output-dir",
        str(output_dir),
        "--num-samples",
        str(num_samples),
    ]
    command.extend(cli_args_from_mapping(eval_args))
    spec = RunSpec(
        run_name=run_name,
        run_type="compare",
        expert_name=expert_name,
        stage_name=stage_name,
        cwd=str(repo_dir),
        output_dir=str(output_dir),
        log_path=str(run_dir / "launch.log"),
        command=command,
        metadata={
            "manifest_source": manifest_source,
            "num_samples": num_samples,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run.json", spec.to_dict())
    write_launch_script(run_dir / "launch.sh", command, repo_dir)
    return spec


def build_packet_run_spec(
    project_config: ProjectConfig,
    track_dir: Path,
    expert_name: str,
    compare_run_name: str,
    stage_name: str,
) -> RunSpec:
    repo_dir = Path(project_config.repo_dir)
    run_name = f"{compare_run_name}_packet"
    run_dir = resolve_run_dir(track_dir, expert_name, run_name)
    compare_dir = resolve_run_dir(track_dir, expert_name, compare_run_name) / "artifacts"
    output_dir = compare_dir / "triptych_packet"
    runtime_python = resolve_runtime_python()
    command = [
        runtime_python,
        str(repo_dir / "build_emoart_compare_triptych_packet.py"),
        "--compare-dir",
        str(compare_dir),
        "--output-dir",
        str(output_dir),
    ]
    spec = RunSpec(
        run_name=run_name,
        run_type="packet",
        expert_name=expert_name,
        stage_name=stage_name,
        cwd=str(repo_dir),
        output_dir=str(output_dir),
        log_path=str(run_dir / "launch.log"),
        command=command,
        metadata={
            "depends_on": compare_run_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run.json", spec.to_dict())
    write_launch_script(run_dir / "launch.sh", command, repo_dir)
    return spec


def build_track_readme(
    project_config: ProjectConfig,
    track_config: TrackConfig,
    track_dir: Path,
    family_summary: Dict[str, Any],
    texture_summary: Dict[str, Any],
    run_specs: List[RunSpec],
) -> str:
    launcher = Path(project_config.repo_dir) / "run_emoart_lab.sh"
    lines: List[str] = []
    lines.append(f"# {track_config.track_name}")
    lines.append("")
    lines.append("## Workflow")
    lines.append("")
    lines.append("1. `prepare` snapshots configs, builds narrow family manifests, builds texture-refine manifests, and materializes isolated train/compare/packet runs.")
    lines.append("2. `stage1_family` trains one narrow expert LoRA against its own `family_train` split and validates on `family_val`.")
    if track_config.continuation is not None:
        queue = " -> ".join(track_config.continuation.expert_order)
        lines.append(
            f"3. The continuation supervisor runs `stage1_family` in expert order: `{queue}`."
        )
        smoke_steps = ", ".join(
            f"`{item.manifest_source}_{item.num_samples}`"
            for item in track_config.continuation.smoke_evaluations
        )
        if smoke_steps:
            lines.append(
                f"4. After each `stage1_family` train, smoke evaluation runs immediately for {smoke_steps}, and each compare run also builds a triptych review packet."
            )
        post_steps = ", ".join(
            f"`{item.manifest_source}_{item.num_samples}`"
            for item in track_config.continuation.post_stage_evaluations
        )
        if post_steps:
            lines.append(
                f"5. After both experts pass the smoke stage, the supervisor runs the standard stage1 evaluation set: {post_steps}, again with packets."
            )
        if not track_config.continuation.auto_advance_to_stage2:
            lines.append("6. The queue stops after stage1 evaluation and does not auto-enter `stage2_texture_refine`.")
    else:
        lines.append("3. `stage2_texture_refine` can resume from `stage1_family` when explicitly launched.")
    lines.append("")
    lines.append("## Architecture")
    lines.append("")
    lines.append("- `configs/`: frozen snapshots of the project and track schema")
    lines.append("- `data/family_manifests/`: narrow expert datasets")
    lines.append("- `data/texture_refine/`: stage-2 texture-rich datasets")
    lines.append("- `runs/<expert>/<run>/`: one isolated unit of execution")
    lines.append("- `system/continuation/`: supervisor state, smoke-check outputs, and detached process metadata")
    lines.append("")
    lines.append("## Experts")
    lines.append("")
    for expert_name, expert in track_config.experts.items():
        family_info = family_summary["experts"][expert_name]
        texture_info = texture_summary["experts"][expert_name]
        lines.append(f"- `{expert_name}`: {', '.join(expert.styles)}")
        lines.append(
            f"  stage1 train={family_info['train_count']}, val={family_info['val_count']}, stage2 train={texture_info['texture_refine_count']}"
        )
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append(
        f"- prepare: `{launcher} prepare --project-config {project_config.snapshot_path} --track-config {track_config.snapshot_path}`"
    )
    lines.append(f"- list runs: `{launcher} list-runs --track-dir {track_dir}`")
    if track_config.continuation is not None:
        lines.append(f"- continue stage1 queue: `{launcher} continue-track --track-dir {track_dir} --background`")
    first_train = next((spec for spec in run_specs if spec.run_type == "train"), None)
    if first_train is not None:
        lines.append(
            f"- start first train: `{launcher} launch --track-dir {track_dir} --expert {first_train.expert_name} --run-name {first_train.run_name} --background`"
        )
    lines.append("")
    lines.append("## Inspection")
    lines.append("")
    lines.append(f"- continuation state: `{track_dir / 'system' / 'continuation' / 'state.json'}`")
    lines.append(f"- continuation process: `{track_dir / 'system' / 'continuation' / 'process.json'}`")
    lines.append(f"- model smoke summary: `{track_dir / 'system' / 'continuation' / 'model_smoke' / 'summary.json'}`")
    lines.append("- train outputs live under `runs/<expert>/train_<stage>/artifacts/` and include `train_log.jsonl`, `train_config.json`, and `final_adapter/`.")
    lines.append("- compare outputs live under `runs/<expert>/compare_<stage>_<manifest>_<n>/artifacts/` and include `summary.json` and `comparison.jsonl`.")
    lines.append("- review packets live under each compare run at `artifacts/triptych_packet/` and include `review_packet.md`, `manual_review_sheet.md`, `packet_manifest.json`, and `sheets/`.")
    lines.append("")
    lines.append("## References")
    lines.append("")
    for key, value in sorted(project_config.references.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Materialized Runs")
    lines.append("")
    for spec in run_specs:
        lines.append(f"- `{spec.expert_name}` / `{spec.run_name}` / `{spec.run_type}`")
    return "\n".join(lines) + "\n"


def prepare_track(project_config_path: Path, track_config_path: Path) -> Dict[str, Any]:
    project_config = ProjectConfig.from_dict(load_json(project_config_path))
    track_config = TrackConfig.from_dict(load_json(track_config_path))
    track_config.validate()

    track_dir = resolve_track_dir(project_config, track_config)
    track_dir.mkdir(parents=True, exist_ok=True)

    config_dir = track_dir / "configs"
    project_config.snapshot_path = str(config_dir / "project_snapshot.json")
    track_config.snapshot_path = str(config_dir / "track_snapshot.json")
    write_json(Path(project_config.snapshot_path), project_config.to_dict())
    write_json(Path(track_config.snapshot_path), track_config.to_dict())

    family_summary = build_family_manifests(
        train_manifest=Path(project_config.data["base_train_manifest"]),
        val_manifest=Path(project_config.data["base_val_manifest"]),
        experts=track_config.experts,
        output_dir=track_dir / "data" / "family_manifests",
    )
    texture_summary = build_texture_refine_manifests(
        experts=track_config.experts,
        track_dir=track_dir,
        texture_rich_manifest=Path(project_config.data["texture_rich_manifest"]),
    )

    run_specs: List[RunSpec] = []
    for expert_name in track_config.experts:
        for stage in track_config.stages:
            run_specs.append(build_train_run_spec(project_config, track_config, track_dir, expert_name, stage))
        for stage_name in track_config.evaluation_matrix.stages:
            for manifest_source in track_config.evaluation_matrix.manifest_sources:
                for num_samples in track_config.evaluation_matrix.sample_counts:
                    compare_spec = build_compare_run_spec(
                        project_config,
                        track_config,
                        track_dir,
                        expert_name,
                        stage_name,
                        manifest_source,
                        num_samples,
                    )
                    run_specs.append(compare_spec)
                    if track_config.evaluation_matrix.build_packet:
                        run_specs.append(
                            build_packet_run_spec(
                                project_config,
                                track_dir,
                                expert_name,
                                compare_spec.run_name,
                                stage_name,
                            )
                        )

    track_index = {
        "track_name": track_config.track_name,
        "track_dir": str(track_dir),
        "model": track_config.model.to_dict(),
        "experts": {name: expert.to_dict() for name, expert in track_config.experts.items()},
        "stages": [stage.to_dict() for stage in track_config.stages],
        "continuation": track_config.continuation.to_dict() if track_config.continuation is not None else None,
        "family_summary_path": str(track_dir / "data" / "family_manifests" / "family_summary.json"),
        "texture_summary_path": str(track_dir / "data" / "texture_refine" / "texture_refine_summary.json"),
        "runs": [spec.to_dict() for spec in run_specs],
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(track_dir / "track_index.json", track_index)
    (track_dir / "README.md").write_text(
        build_track_readme(project_config, track_config, track_dir, family_summary, texture_summary, run_specs),
        encoding="utf-8",
    )
    return track_index
