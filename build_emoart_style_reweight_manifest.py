import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List


STYLE_WEIGHT_PRESETS = {
    "v1": {
        "upsample": {
            "Art Nouveau (Modern)": 1.5,
            "Pointillism": 1.5,
            "Regionalism": 1.4,
            "High Renaissance": 1.3,
            "Color Field Painting": 1.4,
            "Naturalism": 1.3,
            "Orientalism": 1.2,
        },
        "downsample": {
            "China_images": 0.6,
            "Islamic_image": 0.6,
            "Sōsaku hanga": 0.7,
            "Shin-hanga": 0.75,
            "Ukiyo-e": 0.75,
            "Op Art": 0.7,
            "Constructivism": 0.7,
            "Concretism": 0.7,
            "Korea": 0.8,
        },
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a style-reweighted EmoArt training manifest.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--preset", choices=tuple(STYLE_WEIGHT_PRESETS.keys()), default="v1")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def style_name(record: Dict) -> str:
    return record.get("style_name") or record["request_id"].rsplit("_request-", 1)[0]


def target_count(original_count: int, weight: float) -> int:
    return max(1, int(round(original_count * weight)))


def main():
    args = parse_args()
    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)
    summary_path = Path(args.summary_path)
    preset = STYLE_WEIGHT_PRESETS[args.preset]

    rows = load_jsonl(input_path)
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(style_name(row), []).append(row)

    output_rows: List[Dict] = []
    style_summary = []
    for style, items in sorted(grouped.items()):
        weight = 1.0
        if style in preset["upsample"]:
            weight = preset["upsample"][style]
        if style in preset["downsample"]:
            weight = preset["downsample"][style]

        desired = target_count(len(items), weight)

        if desired <= len(items):
            selected = items[:desired]
        else:
            repeats = []
            while len(repeats) + len(items) < desired:
                repeats.extend(items)
            selected = items + repeats[: desired - len(items)]

        output_rows.extend(selected)
        style_summary.append(
            {
                "style_name": style,
                "original_count": len(items),
                "weight": weight,
                "output_count": len(selected),
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_manifest": str(input_path),
        "output_manifest": str(output_path),
        "preset": args.preset,
        "input_count": len(rows),
        "output_count": len(output_rows),
        "upsample": preset["upsample"],
        "downsample": preset["downsample"],
        "style_summary": style_summary,
        "top_output_style_counts": Counter(style_name(row) for row in output_rows).most_common(20),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
