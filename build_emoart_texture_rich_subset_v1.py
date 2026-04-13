import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set


STRONG_TEXTURE_TAGS = {
    "visible brushstrokes",
    "layered paint texture",
    "dense dotted pigment",
    "micro mark clusters",
    "irregular painterly edges",
    "material brushwork depth",
}

MEDIUM_TEXTURE_TAGS = {
    "textured paint handling",
    "paint surface texture",
    "subtle paint surface variation",
    "fine-grained surface detail",
    "micro texture variation",
}

STRONG_MEDIUM_TAGS = {
    "dense mark making",
    "pigment dots",
    "stippled pigment",
    "canvas texture",
    "painted texture",
    "layered surface depth",
}

LINE_RISK_STYLES = {
    "Art Nouveau (Modern)",
    "Pointillism",
    "Regionalism",
    "High Renaissance",
    "Early Renaissance",
    "Mannerism (Late Renaissance)",
}

LOW_DETAIL_PHRASES = [
    "fine and delicate",
    "fine, delicate",
    "smooth and defined",
    "smooth and refined",
    "soft and smooth",
    "thin and smooth",
]

HIGH_DETAIL_PHRASES = [
    "rough",
    "textured",
    "stipple",
    "dense",
    "dotted",
    "irregular",
    "material",
    "layered",
    "organic edge",
    "broken",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build a texture-rich subset manifest from EmoArt generation training data.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--min-score", type=float, default=3.0)
    parser.add_argument("--max-frame-risk", type=float, default=0.55)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_list(values) -> List[str]:
    if not values:
        return []
    return [str(v).strip().lower() for v in values if str(v).strip()]


def contains_any(text: str, phrases: List[str]) -> bool:
    text = (text or "").strip().lower()
    return any(phrase in text for phrase in phrases)


def score_record(row: Dict) -> Dict:
    texture_tags = normalize_list(row.get("texture_tags"))
    medium_tags = normalize_list(row.get("medium_tags"))
    brushstroke = str(row.get("brushstroke_text") or "")
    line_quality = str(row.get("line_quality_text") or "")
    style_name = row.get("style_name", "UNKNOWN")
    image_stats = row.get("image_stats") or {}
    frame_risk = float(image_stats.get("frame_risk_score", 0.0))

    score = 0.0
    reasons: List[str] = []

    strong_texture_hits = sorted(set(texture_tags) & STRONG_TEXTURE_TAGS)
    medium_texture_hits = sorted(set(texture_tags) & MEDIUM_TEXTURE_TAGS)
    strong_medium_hits = sorted(set(medium_tags) & STRONG_MEDIUM_TAGS)

    if strong_texture_hits:
        score += 1.2 * len(strong_texture_hits)
        reasons.append(f"strong_texture:{','.join(strong_texture_hits)}")
    if medium_texture_hits:
        score += 0.5 * len(medium_texture_hits)
        reasons.append(f"medium_texture:{','.join(medium_texture_hits)}")
    if strong_medium_hits:
        score += 0.75 * len(strong_medium_hits)
        reasons.append(f"medium_tags:{','.join(strong_medium_hits)}")

    if contains_any(brushstroke, HIGH_DETAIL_PHRASES):
        score += 1.2
        reasons.append("brushstroke_high_detail")
    if contains_any(line_quality, HIGH_DETAIL_PHRASES):
        score += 0.6
        reasons.append("line_quality_high_detail")

    if contains_any(brushstroke, LOW_DETAIL_PHRASES):
        score -= 0.8
        reasons.append("brushstroke_low_detail")
    if contains_any(line_quality, LOW_DETAIL_PHRASES):
        score -= 0.5
        reasons.append("line_quality_low_detail")

    if style_name in LINE_RISK_STYLES:
        score += 0.8
        reasons.append("style_boost")

    if frame_risk >= 0.40:
        penalty = min(1.5, (frame_risk - 0.40) * 4.0)
        score -= penalty
        reasons.append(f"frame_penalty:{frame_risk:.3f}")

    return {
        "texture_rich_score": round(score, 4),
        "texture_rich_reasons": reasons,
        "frame_risk_score": frame_risk,
    }


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.input_manifest))

    kept_rows: List[Dict] = []
    score_counter = Counter()
    style_counter = Counter()

    for row in rows:
        scored = score_record(row)
        score = scored["texture_rich_score"]
        frame_risk = scored["frame_risk_score"]
        if score < args.min_score:
            continue
        if frame_risk > args.max_frame_risk:
            continue

        tagged = dict(row)
        tagged.update(scored)
        kept_rows.append(tagged)
        style_counter[tagged.get("style_name", "UNKNOWN")] += 1
        score_counter[int(score)] += 1

    kept_rows.sort(key=lambda r: (r["texture_rich_score"], -r["frame_risk_score"]), reverse=True)
    write_jsonl(Path(args.output_manifest), kept_rows)

    summary = {
        "input_manifest": args.input_manifest,
        "output_manifest": args.output_manifest,
        "num_input_rows": len(rows),
        "num_kept_rows": len(kept_rows),
        "keep_fraction": len(kept_rows) / max(len(rows), 1),
        "min_score": args.min_score,
        "max_frame_risk": args.max_frame_risk,
        "top_styles": style_counter.most_common(24),
        "score_histogram_floor": dict(sorted(score_counter.items())),
        "top_examples": [
            {
                "request_id": row.get("request_id"),
                "style_name": row.get("style_name"),
                "texture_rich_score": row.get("texture_rich_score"),
                "frame_risk_score": row.get("frame_risk_score"),
                "texture_rich_reasons": row.get("texture_rich_reasons"),
            }
            for row in kept_rows[:20]
        ],
    }
    Path(args.summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
