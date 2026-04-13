import argparse
import json
from pathlib import Path

from emoart_lab.continuation import continue_prepared_track, launch_continuation_supervisor
from emoart_lab.download import download_model_snapshot
from emoart_lab.launcher import launch_run, read_track_index
from emoart_lab.materialize import prepare_track


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Organized EmoArt experiment launcher.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build manifests and materialize isolated run specs.")
    prepare_parser.add_argument("--project-config", required=True)
    prepare_parser.add_argument("--track-config", required=True)

    list_parser = subparsers.add_parser("list-runs", help="Show all materialized runs for a prepared track.")
    list_parser.add_argument("--track-dir", required=True)

    launch_parser = subparsers.add_parser("launch", help="Launch a materialized run.")
    launch_parser.add_argument("--track-dir", required=True)
    launch_parser.add_argument("--expert", required=True)
    launch_parser.add_argument("--run-name", required=True)
    launch_parser.add_argument("--background", action="store_true", default=False)

    continue_parser = subparsers.add_parser("continue-track", help="Resume the prepared stage1 queue for a track.")
    continue_parser.add_argument("--track-dir", required=True)
    continue_parser.add_argument("--background", action="store_true", default=False)
    continue_parser.add_argument("--worker", action="store_true", default=False, help=argparse.SUPPRESS)

    download_parser = subparsers.add_parser("download-model", help="Download a model snapshot to a local directory.")
    download_parser.add_argument("--repo-id", required=True)
    download_parser.add_argument("--output-dir", required=True)
    download_parser.add_argument("--endpoint", default="https://hf-mirror.com")
    download_parser.add_argument("--revision", default="main")
    download_parser.add_argument("--timeout", type=int, default=30)
    download_parser.add_argument("--weight-parallelism", type=int, default=2)
    download_parser.add_argument("--max-retries", type=int, default=12)
    download_parser.add_argument("--include-readme", action="store_true", default=False)
    download_parser.add_argument("--ignore-env-proxy", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare":
        track_index = prepare_track(Path(args.project_config), Path(args.track_config))
        print(json.dumps(track_index, ensure_ascii=False, indent=2))
        return

    if args.command == "list-runs":
        track_index = read_track_index(Path(args.track_dir))
        print(json.dumps(track_index, ensure_ascii=False, indent=2))
        return

    if args.command == "launch":
        status = launch_run(
            track_dir=Path(args.track_dir),
            expert_name=args.expert,
            run_name=args.run_name,
            background=args.background,
        )
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    if args.command == "continue-track":
        if args.background and args.worker:
            raise RuntimeError("--background and --worker cannot be used together.")
        if args.background:
            status = launch_continuation_supervisor(Path(args.track_dir))
        else:
            status = continue_prepared_track(Path(args.track_dir))
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return

    if args.command == "download-model":
        summary = download_model_snapshot(
            endpoint=args.endpoint,
            repo_id=args.repo_id,
            output_dir=Path(args.output_dir),
            revision=args.revision,
            timeout=args.timeout,
            include_readme=args.include_readme,
            weight_parallelism=args.weight_parallelism,
            max_retries=args.max_retries,
            trust_env_proxy=not args.ignore_env_proxy,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
