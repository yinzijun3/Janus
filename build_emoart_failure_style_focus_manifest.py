import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


FOCUS_PRESETS = {
    "v1": {
        "focus_styles": [
            "Art Nouveau (Modern)",
            "Pointillism",
            "Regionalism",
            "High Renaissance",
            "Color Field Painting",
            "Orientalism",
        ],
        "exclude_anchor_styles": [
            "China_images",
            "Islamic_image",
            "Sōsaku hanga",
            "Shin-hanga",
            "Ukiyo-e",
            "Op Art",
            "Constructivism",
            "Concretism",
            "Korea",
        ],
        "anchor_per_style": 8,
        "focus_repeat": {
            "Art Nouveau (Modern)": 1.35,
            "Pointillism": 1.3,
            "Regionalism": 1.25,
            "High Renaissance": 1.2,
            "Color Field Painting": 1.2,
            "Orientalism": 1.15,
        },
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a focused failure-style continuation manifest for EmoArt.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--preset", choices=tuple(FOCUS_PRESETS.keys()), default="v1")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def style_name(record: Dict) -> str:
    return record.get("style_name") or record["request_id"].rsplit("_request-", 1)[0]


def repeat_rows(rows: List[Dict], weight: float) -> List[Dict]:
    if weight <= 1.0:
        return list(rows)
    desired = max(len(rows), int(round(len(rows) * weight)))
    output = list(rows)
    while len(output) < desired:
        output.extend(rows[: desired - len(output)])
    return output


def main():
    args = parse_args()
    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)
    summary_path = Path(args.summary_path)
    preset = FOCUS_PRESETS[args.preset]

    rng = random.Random(args.seed)
    rows = load_jsonl(input_path)

    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(style_name(row), []).append(row)

    focus_styles = set(preset["focus_styles"])
    excluded_anchor_styles = set(preset["exclude_anchor_styles"])
    anchor_per_style = int(preset["anchor_per_style"])

    output_rows: List[Dict] = []
    style_summary = []

    for style in preset["focus_styles"]:
        items = list(grouped.get(style, []))
        weighted = repeat_rows(items, preset["focus_repeat"].get(style, 1.0))
        output_rows.extend(weighted)
        style_summary.append(
            {
                "style_name": style,
                "kind": "focus",
                "original_count": len(items),
                "output_count": len(weighted),
                "weight": preset["focus_repeat"].get(style, 1.0),
            }
        )

    anchor_styles = []
    for style, items in sorted(grouped.items()):
        if style in focus_styles or style in excluded_anchor_styles:
            continue
        anchor_styles.append(style)
        if len(items) <= anchor_per_style:
            selected = list(items)
        else:
            selected = rng.sample(items, anchor_per_style)
            selected.sort(key=lambda row: row["request_id"])
        output_rows.extend(selected)
        style_summary.append(
            {
                "style_name": style,
                "kind": "anchor",
                "original_count": len(items),
                "output_count": len(selected),
                "weight": round(len(selected) / max(len(items), 1), 4),
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_manifest": str(input_path),
        "output_manifest": str(output_path),
        "preset": args.preset,
        "seed": args.seed,
        "input_count": len(rows),
        "output_count": len(output_rows),
        "focus_styles": preset["focus_styles"],
        "exclude_anchor_styles": preset["exclude_anchor_styles"],
        "anchor_per_style": anchor_per_style,
        "focus_repeat": preset["focus_repeat"],
        "num_focus_styles": len(preset["focus_styles"]),
        "num_anchor_styles": len(anchor_styles),
        "style_summary": style_summary,
        "top_output_style_counts": Counter(style_name(row) for row in output_rows).most_common(20),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
