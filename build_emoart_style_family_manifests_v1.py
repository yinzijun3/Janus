import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List


FAMILY_TO_STYLES: Dict[str, List[str]] = {
    "line_structure": [
        "Art Nouveau (Modern)",
        "Early Renaissance",
        "High Renaissance",
        "Mannerism (Late Renaissance)",
        "Pointillism",
        "Regionalism",
    ],
    "east_asian_scroll": [
        "China_images",
        "Gongbi",
        "Ink and wash painting",
        "Islamic_image",
        "Korea",
        "Shin-hanga",
        "Sōsaku hanga",
        "Ukiyo-e",
    ],
    "classical_figurative": [
        "Academicism",
        "Baroque",
        "Contemporary Realism",
        "Impressionism",
        "Magic Realism",
        "Naturalism",
        "Neo-Romanticism",
        "New Realism",
        "Orientalism",
        "Post-Impressionism",
        "Realism",
        "Rococo",
        "Romanticism",
        "Social Realism",
        "Socialist Realism",
    ],
    "expressive_painterly": [
        "Abstract Art",
        "Abstract Expressionism",
        "Art Brut",
        "Art Informel",
        "Dada",
        "Expressionism",
        "Fauvism",
        "Figurative Expressionism",
        "Lyrical Abstraction",
        "Neo-Expressionism",
        "Surrealism",
        "Symbolism",
        "Tachisme",
        "Transavantgarde",
    ],
    "graphic_flat": [
        "Art Deco",
        "Color Field Painting",
        "Concretism",
        "Constructivism",
        "Cubism",
        "Hard Edge Painting",
        "India_art",
        "Native Art",
        "Naïve Art (Primitivism)",
        "Neo-Impressionism",
        "Neo-Pop Art",
        "Op Art",
        "Pop Art",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-path", required=True)
    return parser.parse_args()


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    style_to_family = {}
    for family, styles in FAMILY_TO_STYLES.items():
        for style in styles:
            if style in style_to_family:
                raise ValueError(f"Duplicate style assignment: {style}")
            style_to_family[style] = family

    train_rows = load_jsonl(args.train_manifest)
    val_rows = load_jsonl(args.val_manifest)

    seen_styles = set()
    for row in train_rows + val_rows:
        seen_styles.add(row.get("style_name", "UNKNOWN"))
    missing_styles = sorted(seen_styles - set(style_to_family))
    if missing_styles:
        raise ValueError(f"Unassigned styles: {missing_styles}")

    split_rows = {"train": train_rows, "val": val_rows}
    summary = {
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "output_dir": args.output_dir,
        "families": {},
        "style_to_family": style_to_family,
    }

    for family in FAMILY_TO_STYLES:
        summary["families"][family] = {
            "styles": FAMILY_TO_STYLES[family],
            "train_count": 0,
            "val_count": 0,
            "train_style_counts": {},
            "val_style_counts": {},
        }

    for split_name, rows in split_rows.items():
        family_buckets = defaultdict(list)
        style_counters = defaultdict(Counter)
        for row in rows:
            style_name = row.get("style_name", "UNKNOWN")
            family = style_to_family[style_name]
            tagged = dict(row)
            tagged["style_family"] = family
            family_buckets[family].append(tagged)
            style_counters[family][style_name] += 1

        for family, family_rows in family_buckets.items():
            out_path = os.path.join(args.output_dir, f"{family}_{split_name}.jsonl")
            write_jsonl(out_path, family_rows)
            summary["families"][family][f"{split_name}_count"] = len(family_rows)
            summary["families"][family][f"{split_name}_style_counts"] = dict(
                sorted(style_counters[family].items())
            )

    with open(os.path.join(args.output_dir, "style_family_map_v1.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "family_to_styles": FAMILY_TO_STYLES,
                "style_to_family": style_to_family,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
