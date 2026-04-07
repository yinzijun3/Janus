import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Build a repeated prompt-only retention manifest for EmoArt.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--repeat-factor", type=int, default=8)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    args = parse_args()
    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)
    summary_path = Path(args.summary_path)

    rows = load_jsonl(input_path)
    output_rows: List[Dict] = []
    for repeat_index in range(args.repeat_factor):
        for row in rows:
            repeated = dict(row)
            source_id = row.get("id") or f"prompt_{len(output_rows):04d}"
            repeated["id"] = f"{source_id}__rep{repeat_index:02d}"
            repeated["source_id"] = source_id
            repeated["repeat_index"] = repeat_index
            output_rows.append(repeated)

    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_manifest": str(input_path),
        "output_manifest": str(output_path),
        "repeat_factor": args.repeat_factor,
        "input_count": len(rows),
        "output_count": len(output_rows),
        "category_counts": Counter(row.get("category", "<none>") for row in output_rows),
        "source_prompt_count": len({row.get("source_id", row.get("id")) for row in output_rows}),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
