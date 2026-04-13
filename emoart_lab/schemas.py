from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


MANIFEST_SOURCES = {"family_train", "family_val", "texture_train", "full_train", "full_val"}


@dataclass
class ModelDownloadConfig:
    repo_id: str
    endpoint: str
    revision: str = "main"
    weight_parallelism: int = 2
    max_retries: int = 20
    ignore_env_proxy: bool = False
    expected_sizes: Dict[str, int] = field(default_factory=dict)
    required_files: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelDownloadConfig":
        return cls(
            repo_id=payload["repo_id"],
            endpoint=payload["endpoint"],
            revision=payload.get("revision", "main"),
            weight_parallelism=int(payload.get("weight_parallelism", 2)),
            max_retries=int(payload.get("max_retries", 20)),
            ignore_env_proxy=bool(payload.get("ignore_env_proxy", False)),
            expected_sizes={key: int(value) for key, value in payload.get("expected_sizes", {}).items()},
            required_files=list(payload.get("required_files", [])),
        )

    def validate(self) -> None:
        if self.weight_parallelism <= 0:
            raise ValueError(f"weight_parallelism must be positive, got {self.weight_parallelism}.")
        if self.max_retries <= 0:
            raise ValueError(f"max_retries must be positive, got {self.max_retries}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "endpoint": self.endpoint,
            "revision": self.revision,
            "weight_parallelism": self.weight_parallelism,
            "max_retries": self.max_retries,
            "ignore_env_proxy": self.ignore_env_proxy,
            "expected_sizes": self.expected_sizes,
            "required_files": self.required_files,
        }


@dataclass
class ProjectConfig:
    project_name: str
    repo_dir: str
    output_root: str
    data: Dict[str, str]
    references: Dict[str, str]
    snapshot_path: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProjectConfig":
        return cls(
            project_name=payload["project_name"],
            repo_dir=payload["repo_dir"],
            output_root=payload["output_root"],
            data=dict(payload.get("data", {})),
            references=dict(payload.get("references", {})),
            snapshot_path=payload.get("_snapshot_path", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "project_name": self.project_name,
            "repo_dir": self.repo_dir,
            "output_root": self.output_root,
            "data": self.data,
            "references": self.references,
        }
        if self.snapshot_path:
            payload["_snapshot_path"] = self.snapshot_path
        return payload


@dataclass
class ModelConfig:
    model_key: str
    model_path: str
    dtype: str
    download: Optional[ModelDownloadConfig] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelConfig":
        return cls(
            model_key=payload["model_key"],
            model_path=payload["model_path"],
            dtype=payload["dtype"],
            download=ModelDownloadConfig.from_dict(payload["download"]) if payload.get("download") else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "model_key": self.model_key,
            "model_path": self.model_path,
            "dtype": self.dtype,
        }
        if self.download is not None:
            payload["download"] = self.download.to_dict()
        return payload


@dataclass
class EvalRunConfig:
    manifest_source: str
    num_samples: int
    build_packet: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvalRunConfig":
        return cls(
            manifest_source=payload["manifest_source"],
            num_samples=int(payload["num_samples"]),
            build_packet=bool(payload.get("build_packet", True)),
        )

    def validate(self) -> None:
        if self.manifest_source not in MANIFEST_SOURCES:
            raise ValueError(f"Unsupported manifest source in continuation config: {self.manifest_source}")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_source": self.manifest_source,
            "num_samples": self.num_samples,
            "build_packet": self.build_packet,
        }


@dataclass
class ContinuationConfig:
    expert_order: List[str]
    train_stage: str
    smoke_evaluations: List[EvalRunConfig]
    post_stage_evaluations: List[EvalRunConfig]
    auto_advance_to_stage2: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ContinuationConfig":
        return cls(
            expert_order=list(payload.get("expert_order", [])),
            train_stage=payload["train_stage"],
            smoke_evaluations=[EvalRunConfig.from_dict(item) for item in payload.get("smoke_evaluations", [])],
            post_stage_evaluations=[EvalRunConfig.from_dict(item) for item in payload.get("post_stage_evaluations", [])],
            auto_advance_to_stage2=bool(payload.get("auto_advance_to_stage2", False)),
        )

    def validate(self, known_experts: List[str], known_stages: List[str]) -> None:
        if self.train_stage not in known_stages:
            raise ValueError(f"Continuation train stage '{self.train_stage}' is not a known stage.")
        for expert_name in self.expert_order:
            if expert_name not in known_experts:
                raise ValueError(f"Continuation expert '{expert_name}' is not defined in the track.")
        if not self.smoke_evaluations:
            raise ValueError("Continuation config requires at least one smoke evaluation.")
        for evaluation in self.smoke_evaluations + self.post_stage_evaluations:
            evaluation.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_order": self.expert_order,
            "train_stage": self.train_stage,
            "smoke_evaluations": [item.to_dict() for item in self.smoke_evaluations],
            "post_stage_evaluations": [item.to_dict() for item in self.post_stage_evaluations],
            "auto_advance_to_stage2": self.auto_advance_to_stage2,
        }


@dataclass
class ExpertConfig:
    name: str
    styles: List[str]
    description: str = ""

    @classmethod
    def from_dict(cls, name: str, payload: Dict[str, Any]) -> "ExpertConfig":
        return cls(
            name=name,
            styles=list(payload.get("styles", [])),
            description=payload.get("description", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "styles": self.styles,
        }


@dataclass
class TrainStageConfig:
    name: str
    train_manifest: str
    val_manifest: str
    train_args: Dict[str, Any]
    resume_from_stage: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainStageConfig":
        return cls(
            name=payload["name"],
            train_manifest=payload["train_manifest"],
            val_manifest=payload["val_manifest"],
            train_args=dict(payload.get("train_args", {})),
            resume_from_stage=payload.get("resume_from_stage"),
        )

    def validate(self, known_stages: List[str]) -> None:
        if self.train_manifest not in MANIFEST_SOURCES:
            raise ValueError(f"Unsupported train manifest source: {self.train_manifest}")
        if self.val_manifest not in MANIFEST_SOURCES:
            raise ValueError(f"Unsupported val manifest source: {self.val_manifest}")
        if self.resume_from_stage and self.resume_from_stage not in known_stages:
            raise ValueError(
                f"Stage '{self.name}' resumes from unknown stage '{self.resume_from_stage}'."
            )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "train_manifest": self.train_manifest,
            "val_manifest": self.val_manifest,
            "train_args": self.train_args,
        }
        if self.resume_from_stage:
            payload["resume_from_stage"] = self.resume_from_stage
        return payload


@dataclass
class EvaluationMatrix:
    stages: List[str]
    manifest_sources: List[str]
    sample_counts: List[int]
    build_packet: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationMatrix":
        return cls(
            stages=list(payload.get("stages", [])),
            manifest_sources=list(payload.get("manifest_sources", [])),
            sample_counts=list(payload.get("sample_counts", [])),
            build_packet=bool(payload.get("build_packet", True)),
        )

    def validate(self, known_stages: List[str]) -> None:
        for stage_name in self.stages:
            if stage_name not in known_stages:
                raise ValueError(f"Evaluation matrix references unknown stage '{stage_name}'.")
        for source_name in self.manifest_sources:
            if source_name not in MANIFEST_SOURCES:
                raise ValueError(f"Evaluation matrix references unsupported manifest source '{source_name}'.")
        for value in self.sample_counts:
            if value <= 0:
                raise ValueError(f"Evaluation sample counts must be positive, got {value}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": self.stages,
            "manifest_sources": self.manifest_sources,
            "sample_counts": self.sample_counts,
            "build_packet": self.build_packet,
        }


@dataclass
class TrackConfig:
    track_name: str
    model: ModelConfig
    experts: Dict[str, ExpertConfig]
    stages: List[TrainStageConfig]
    evaluation_defaults: Dict[str, Any] = field(default_factory=dict)
    evaluation_matrix: EvaluationMatrix = field(default_factory=lambda: EvaluationMatrix([], [], []))
    continuation: Optional[ContinuationConfig] = None
    snapshot_path: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrackConfig":
        experts = {
            name: ExpertConfig.from_dict(name, expert_payload)
            for name, expert_payload in payload.get("experts", {}).items()
        }
        return cls(
            track_name=payload["track_name"],
            model=ModelConfig.from_dict(payload["model"]),
            experts=experts,
            stages=[TrainStageConfig.from_dict(stage_payload) for stage_payload in payload.get("stages", [])],
            evaluation_defaults=dict(payload.get("evaluation_defaults", {})),
            evaluation_matrix=EvaluationMatrix.from_dict(payload.get("evaluation_matrix", {})),
            continuation=ContinuationConfig.from_dict(payload["continuation"]) if payload.get("continuation") else None,
            snapshot_path=payload.get("_snapshot_path", ""),
        )

    def validate(self) -> None:
        style_to_expert: Dict[str, str] = {}
        stage_names: List[str] = []
        for expert_name, expert in self.experts.items():
            if not expert.styles:
                raise ValueError(f"Expert '{expert_name}' has no assigned styles.")
            for style_name in expert.styles:
                if style_name in style_to_expert:
                    raise ValueError(
                        f"Style '{style_name}' is assigned to both '{style_to_expert[style_name]}' and '{expert_name}'."
                    )
                style_to_expert[style_name] = expert_name

        for stage in self.stages:
            if stage.name in stage_names:
                raise ValueError(f"Duplicate stage name: {stage.name}")
            stage.validate(stage_names)
            stage_names.append(stage.name)

        self.evaluation_matrix.validate(stage_names)
        if self.model.download is not None:
            self.model.download.validate()
        if self.continuation is not None:
            self.continuation.validate(list(self.experts.keys()), stage_names)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "track_name": self.track_name,
            "model": self.model.to_dict(),
            "experts": {name: expert.to_dict() for name, expert in self.experts.items()},
            "stages": [stage.to_dict() for stage in self.stages],
            "evaluation_defaults": self.evaluation_defaults,
            "evaluation_matrix": self.evaluation_matrix.to_dict(),
        }
        if self.continuation is not None:
            payload["continuation"] = self.continuation.to_dict()
        if self.snapshot_path:
            payload["_snapshot_path"] = self.snapshot_path
        return payload


@dataclass
class RunSpec:
    run_name: str
    run_type: str
    expert_name: str
    stage_name: str
    cwd: str
    output_dir: str
    log_path: str
    command: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "run_name": self.run_name,
            "run_type": self.run_type,
            "expert_name": self.expert_name,
            "stage_name": self.stage_name,
            "cwd": self.cwd,
            "output_dir": self.output_dir,
            "log_path": self.log_path,
            "command": self.command,
        }
        payload.update(self.metadata)
        return payload
