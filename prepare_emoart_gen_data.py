import argparse
import json
import os

from finetune.emoart_generation import prepare_emoart_generation_manifests


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare EmoArt-5k manifests for Janus image-generation LoRA.")
    parser.add_argument(
        "--annotation-path",
        default="/root/autodl-tmp/data/emoart_5k/annotation.official.json",
    )
    parser.add_argument(
        "--images-root",
        default="/root/autodl-tmp/data/emoart_5k/Images",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-bad-images", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    result = prepare_emoart_generation_manifests(
        annotation_path=args.annotation_path,
        images_root=args.images_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples or None,
        skip_bad_images=args.skip_bad_images,
    )
    print(json.dumps(result["stats"], ensure_ascii=False, indent=2))
    print(f"train_manifest={result['train_path']}")
    print(f"val_manifest={result['val_path']}")
    print(f"audit_summary={result['stats_path']}")
    print(f"bad_samples={result['bad_path']}")


if __name__ == "__main__":
    main()
