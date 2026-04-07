import argparse
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


def parse_args():
    parser = argparse.ArgumentParser(description="Audit EmoArt generation data for frame/border bias and style entropy risks.")
    parser.add_argument("--manifest-path", required=True, help="Path to train/val JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for audit outputs.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for debugging.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--style-sample-size",
        type=int,
        default=24,
        help="Max images sampled per style for perceptual hash diversity estimation.",
    )
    parser.add_argument(
        "--border-ratio",
        type=float,
        default=0.08,
        help="Width of the outer border band as a fraction of min(image_width, image_height).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def safe_style_name(record: Dict) -> str:
    return record.get("style_name") or record["request_id"].rsplit("_request-", 1)[0]


def grayscale_gradients(gray: np.ndarray) -> np.ndarray:
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return dx + dy


def average_hash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    resized = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32)
    threshold = float(arr.mean())
    return (arr >= threshold).astype(np.uint8).reshape(-1)


def analyze_image(image_path: Path, border_ratio: float) -> Dict:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        arr = np.asarray(rgb, dtype=np.float32)
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)
        gradients = grayscale_gradients(gray)

    min_dim = min(width, height)
    border = max(2, int(min_dim * border_ratio))
    if border * 4 >= min_dim:
        border = max(1, min_dim // 8)

    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]
    border_mask = (yy < border) | (yy >= height - border) | (xx < border) | (xx >= width - border)

    center_margin = max(border * 2, min_dim // 5)
    center_top = min(center_margin, height // 2)
    center_left = min(center_margin, width // 2)
    center_bottom = max(center_top + 1, height - center_margin)
    center_right = max(center_left + 1, width - center_margin)
    center_mask = np.zeros((height, width), dtype=bool)
    center_mask[center_top:center_bottom, center_left:center_right] = True

    border_gray = gray[border_mask]
    center_gray = gray[center_mask]
    border_grad = gradients[border_mask]
    center_grad = gradients[center_mask]
    border_rgb = arr[border_mask]
    center_rgb = arr[center_mask]

    border_mean = float(border_gray.mean())
    center_mean = float(center_gray.mean())
    border_std = float(border_gray.std())
    center_std = float(center_gray.std())
    border_grad_mean = float(border_grad.mean())
    center_grad_mean = float(center_grad.mean())
    border_color_std = float(border_rgb.std())
    center_color_std = float(center_rgb.std())

    bright_border_score = clip01((border_mean - center_mean - 8.0) / 32.0)
    uniform_border_score = clip01((26.0 - border_std) / 26.0)
    edge_drop_score = clip01((center_grad_mean - border_grad_mean) / max(center_grad_mean, 1e-6))
    low_chroma_border_score = clip01((28.0 - border_color_std) / 28.0)
    aspect_ratio = width / max(height, 1)
    tall_scroll_score = clip01((1.0 / max(aspect_ratio, 1e-6) - 1.2) / 0.8)

    frame_risk_score = (
        0.30 * bright_border_score
        + 0.25 * uniform_border_score
        + 0.25 * edge_drop_score
        + 0.10 * low_chroma_border_score
        + 0.10 * tall_scroll_score
    )

    ahash = average_hash(rgb)
    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "border_mean_luma": border_mean,
        "center_mean_luma": center_mean,
        "border_luma_std": border_std,
        "center_luma_std": center_std,
        "border_grad_mean": border_grad_mean,
        "center_grad_mean": center_grad_mean,
        "border_color_std": border_color_std,
        "center_color_std": center_color_std,
        "bright_border_score": bright_border_score,
        "uniform_border_score": uniform_border_score,
        "edge_drop_score": edge_drop_score,
        "low_chroma_border_score": low_chroma_border_score,
        "tall_scroll_score": tall_scroll_score,
        "frame_risk_score": frame_risk_score,
        "ahash_bits": ahash.tolist(),
    }


def hamming_distance(bits_a: List[int], bits_b: List[int]) -> int:
    a = np.asarray(bits_a, dtype=np.uint8)
    b = np.asarray(bits_b, dtype=np.uint8)
    return int(np.count_nonzero(a != b))


def estimate_style_entropy(style_rows: List[Dict], sample_size: int, seed: int) -> Dict:
    if not style_rows:
        return {
            "hash_sample_size": 0,
            "mean_hash_distance": None,
            "low_entropy_score": None,
        }

    rng = random.Random(seed)
    sample = list(style_rows)
    if len(sample) > sample_size:
        sample = rng.sample(sample, sample_size)

    hashes = [row["image_stats"]["ahash_bits"] for row in sample if row.get("image_stats")]
    if len(hashes) < 2:
        return {
            "hash_sample_size": len(hashes),
            "mean_hash_distance": None,
            "low_entropy_score": None,
        }

    distances = [hamming_distance(a, b) for a, b in itertools.combinations(hashes, 2)]
    mean_distance = float(np.mean(distances))
    low_entropy_score = clip01(1.0 - (mean_distance / 64.0))
    return {
        "hash_sample_size": len(hashes),
        "mean_hash_distance": mean_distance,
        "low_entropy_score": low_entropy_score,
    }


def summarize_styles(rows: List[Dict], style_sample_size: int, seed: int) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["style_name"]].append(row)

    style_rows: List[Dict] = []
    total_count = sum(len(v) for v in grouped.values())
    max_count = max((len(v) for v in grouped.values()), default=1)

    for style_name, items in grouped.items():
        frame_scores = [item["image_stats"]["frame_risk_score"] for item in items]
        bright_scores = [item["image_stats"]["bright_border_score"] for item in items]
        edge_scores = [item["image_stats"]["edge_drop_score"] for item in items]
        tall_scores = [item["image_stats"]["tall_scroll_score"] for item in items]
        entropy = estimate_style_entropy(items, sample_size=style_sample_size, seed=seed + len(style_name))
        count = len(items)
        count_weight = math.sqrt(count / max_count)
        risk_score = (
            0.45 * float(np.mean(frame_scores))
            + 0.35 * (entropy["low_entropy_score"] or 0.0)
            + 0.20 * count_weight
        )
        top_examples = sorted(items, key=lambda x: x["image_stats"]["frame_risk_score"], reverse=True)[:5]
        style_rows.append(
            {
                "style_name": style_name,
                "count": count,
                "count_fraction": count / max(total_count, 1),
                "avg_frame_risk_score": float(np.mean(frame_scores)),
                "avg_bright_border_score": float(np.mean(bright_scores)),
                "avg_edge_drop_score": float(np.mean(edge_scores)),
                "avg_tall_scroll_score": float(np.mean(tall_scores)),
                "mean_hash_distance": entropy["mean_hash_distance"],
                "low_entropy_score": entropy["low_entropy_score"],
                "hash_sample_size": entropy["hash_sample_size"],
                "style_risk_score": risk_score,
                "top_examples": [
                    {
                        "request_id": row["request_id"],
                        "image_path": row["image_path"],
                        "frame_risk_score": row["image_stats"]["frame_risk_score"],
                    }
                    for row in top_examples
                ],
            }
        )

    style_rows.sort(key=lambda row: row["style_risk_score"], reverse=True)
    return style_rows


def build_report(summary: Dict, style_rows: List[Dict], sample_rows: List[Dict]) -> str:
    lines = [
        "# EmoArt Data Risk Audit",
        "",
        "## Summary",
        f"- `num_records`: {summary['num_records']}",
        f"- `num_styles`: {summary['num_styles']}",
        f"- `mean_frame_risk_score`: {summary['mean_frame_risk_score']:.4f}",
        f"- `high_frame_risk_count(>=0.55)`: {summary['high_frame_risk_count']}",
        f"- `very_high_frame_risk_count(>=0.70)`: {summary['very_high_frame_risk_count']}",
        "",
        "## Highest-Risk Styles",
    ]
    for row in style_rows[:15]:
        mean_hash = "n/a" if row["mean_hash_distance"] is None else f"{row['mean_hash_distance']:.2f}"
        low_entropy = "n/a" if row["low_entropy_score"] is None else f"{row['low_entropy_score']:.3f}"
        lines.append(
            f"- `{row['style_name']}`: risk `{row['style_risk_score']:.3f}`, count `{row['count']}`, "
            f"avg_frame `{row['avg_frame_risk_score']:.3f}`, low_entropy `{low_entropy}`, mean_hash_distance `{mean_hash}`"
        )

    lines.extend(["", "## Highest-Risk Samples"])
    for row in sample_rows[:25]:
        stats = row["image_stats"]
        lines.append(
            f"- `{row['request_id']}` ({row['style_name']}): frame `{stats['frame_risk_score']:.3f}`, "
            f"bright_border `{stats['bright_border_score']:.3f}`, edge_drop `{stats['edge_drop_score']:.3f}`, "
            f"scroll `{stats['tall_scroll_score']:.3f}`"
        )

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(manifest_path)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    analyzed_rows: List[Dict] = []
    failures: List[Dict] = []
    for row in rows:
        style_name = safe_style_name(row)
        image_path = Path(row["image_path"])
        try:
            image_stats = analyze_image(image_path, border_ratio=args.border_ratio)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            failures.append(
                {
                    "request_id": row.get("request_id", ""),
                    "style_name": style_name,
                    "image_path": str(image_path),
                    "error": str(exc),
                }
            )
            continue

        analyzed_rows.append(
            {
                "request_id": row["request_id"],
                "style_name": style_name,
                "image_path": str(image_path),
                "prompt": row.get("prompt", ""),
                "texture_tags": row.get("texture_tags", []),
                "medium_tags": row.get("medium_tags", []),
                "image_stats": image_stats,
            }
        )

    style_rows = summarize_styles(analyzed_rows, style_sample_size=args.style_sample_size, seed=args.seed)
    sample_rows = sorted(analyzed_rows, key=lambda row: row["image_stats"]["frame_risk_score"], reverse=True)

    frame_scores = [row["image_stats"]["frame_risk_score"] for row in analyzed_rows]
    summary = {
        "manifest_path": str(manifest_path),
        "num_records": len(analyzed_rows),
        "num_failures": len(failures),
        "num_styles": len({row["style_name"] for row in analyzed_rows}),
        "mean_frame_risk_score": float(np.mean(frame_scores)) if frame_scores else 0.0,
        "median_frame_risk_score": float(np.median(frame_scores)) if frame_scores else 0.0,
        "high_frame_risk_count": sum(score >= 0.55 for score in frame_scores),
        "very_high_frame_risk_count": sum(score >= 0.70 for score in frame_scores),
        "top_style_counts": Counter(row["style_name"] for row in analyzed_rows).most_common(20),
    }

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "style_summary.json", style_rows)
    write_jsonl(output_dir / "high_frame_risk_samples.jsonl", sample_rows[:250])
    write_json(output_dir / "failures.json", failures)
    (output_dir / "report.md").write_text(build_report(summary, style_rows, sample_rows), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
