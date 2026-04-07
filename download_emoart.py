import argparse
import json
import os
import shutil
import subprocess

import requests

from finetune.emoart import prepare_emoart_manifests

DEFAULT_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
DATASET_REPO = "printblue/EmoArt-5k"


def run_command(command):
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def ensure_annotation(dataset_root: str) -> str:
    annotation_path = os.path.join(dataset_root, "annotation.json")
    if os.path.exists(annotation_path):
        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                json.load(f)
            return annotation_path
        except json.JSONDecodeError:
            os.remove(annotation_path)

    try:
        run_command(
            [
                "curl",
                "-L",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "-o",
                annotation_path,
                f"{DEFAULT_HF_ENDPOINT}/datasets/{DATASET_REPO}/resolve/main/annotation.json",
            ]
        )
        with open(annotation_path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        print("Falling back to datasets-server rows API for annotation metadata.")
        rows = []
        offset = 0
        page_size = 100
        total_rows = None
        while total_rows is None or offset < total_rows:
            response = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                "dataset": "printblue/EmoArt-5k",
                    # datasets-server does not expose a mirror endpoint; keep this as fallback only.
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": page_size,
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            total_rows = payload["num_rows_total"]
            rows.extend(item["row"] for item in payload["rows"])
            offset += page_size
            print(f"Fetched metadata rows: {min(offset, total_rows)}/{total_rows}")

        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return annotation_path


def ensure_images_archive(dataset_root: str) -> str:
    archive_path = os.path.join(dataset_root, "Images.tar.gz")
    if os.path.exists(archive_path):
        return archive_path

    downloader = shutil.which("aria2c")
    if downloader:
        headers = subprocess.check_output(
            [
                "curl",
                "-sI",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                f"{DEFAULT_HF_ENDPOINT}/datasets/{DATASET_REPO}/resolve/main/Images.tar.gz",
            ],
            text=True,
        )
        location = ""
        for line in headers.splitlines():
            if line.lower().startswith("location:"):
                location = line.split(":", 1)[1].strip()
                break
        if not location:
            raise RuntimeError("Failed to resolve direct download URL for Images.tar.gz")

        run_command(
            [
                downloader,
                "-x",
                "16",
                "-s",
                "16",
                "-k",
                "1M",
                "-c",
                "-d",
                dataset_root,
                "-o",
                "Images.tar.gz",
                location,
            ]
        )
    else:
        run_command(
            [
                "curl",
                "-L",
                "-C",
                "-",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "-o",
                archive_path,
                f"{DEFAULT_HF_ENDPOINT}/datasets/{DATASET_REPO}/resolve/main/Images.tar.gz",
            ]
        )
    return archive_path


def ensure_extracted_images(dataset_root: str, archive_path: str) -> str:
    images_root = os.path.join(dataset_root, "Images")
    if os.path.isdir(images_root):
        return images_root

    run_command(["tar", "-xzf", archive_path, "-C", dataset_root])
    return images_root


def parse_args():
    parser = argparse.ArgumentParser(description="Download and audit the EmoArt-5k dataset.")
    parser.add_argument(
        "--dataset-root",
        default="/root/autodl-tmp/data/emoart_5k",
        help="Local directory used for the dataset files.",
    )
    parser.add_argument(
        "--manifest-dir",
        default="/root/autodl-tmp/data/emoart_5k/manifests",
        help="Directory used for train/val manifests and audit logs.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--keep-bad-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.dataset_root, exist_ok=True)
    os.makedirs(args.manifest_dir, exist_ok=True)

    annotation_path = ensure_annotation(args.dataset_root)
    archive_path = ensure_images_archive(args.dataset_root)
    images_root = ensure_extracted_images(args.dataset_root, archive_path)

    manifest_info = prepare_emoart_manifests(
        annotation_path=annotation_path,
        images_root=images_root,
        output_dir=args.manifest_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
        skip_bad_images=not args.keep_bad_images,
    )

    print("Dataset root:", args.dataset_root)
    print("HF endpoint:", DEFAULT_HF_ENDPOINT)
    print("Annotation:", annotation_path)
    print("Images root:", images_root)
    print("Train manifest:", manifest_info["train_path"])
    print("Val manifest:", manifest_info["val_path"])
    print("Audit summary:", manifest_info["stats_path"])
    print("Bad samples:", manifest_info["bad_path"])
    print("Stats:", manifest_info["stats"])


if __name__ == "__main__":
    main()
