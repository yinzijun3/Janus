import argparse
import json
import textwrap
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


VARIANT_ORDER = ["target", "baseline", "prob035", "prob015", "brushstroke"]
VARIANT_DISPLAY = {
    "target": "target",
    "baseline": "baseline",
    "prob035": "head prob0.35",
    "prob015": "head prob0.15",
    "brushstroke": "head prob0.15 brush",
}

CELL_SIZE = 256
PADDING = 16
HEADER_HEIGHT = 126
CAPTION_HEIGHT = 66
BACKGROUND = (248, 246, 241)
TEXT = (20, 20, 20)
MUTED = (90, 90, 90)
BORDER = (210, 205, 198)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ascii_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii")


def metric_line(row: Dict[str, Any]) -> List[str]:
    return [
        (
            f"nll {row['reference_token_nll_after']:.3f}  "
            f"psnr {row['reference_psnr_after']:.2f}  "
            f"mae {row['reference_mae_after']:.1f}"
        ),
        (
            f"g {row['reference_gradient_energy_abs_error_after']:.1f}  "
            f"l {row['reference_laplacian_variance_abs_error_after']:.0f}  "
            f"h {row['reference_high_frequency_energy_abs_error_after']:.0f}"
        ),
    ]


def short_tags(tags: List[str], max_items: int = 4) -> str:
    if not tags:
        return "-"
    if len(tags) <= max_items:
        return ", ".join(tags)
    return ", ".join(tags[:max_items]) + f" (+{len(tags) - max_items})"


def open_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    width_chars: int,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_spacing: int = 2,
) -> int:
    x, y = xy
    lines = textwrap.wrap(text, width=width_chars) or [text]
    line_height = 11 + line_spacing
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def make_sheet(
    item: Dict[str, Any],
    variant_rows: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    width = PADDING * 2 + len(VARIANT_ORDER) * CELL_SIZE + (len(VARIANT_ORDER) - 1) * PADDING
    height = HEADER_HEIGHT + CELL_SIZE + CAPTION_HEIGHT + PADDING * 2
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    request_id = item["request_id"]
    style_name = item["style_name"]
    header_1 = f"{ascii_text(request_id)} | {ascii_text(style_name)}"
    header_2 = (
        f"section={item['section']} | best={item.get('best_variant_vs_baseline', '-')} | "
        f"reason={ascii_text(item['reason'])}"
    )
    header_3 = (
        f"texture_tags: {ascii_text(short_tags(item.get('texture_tags', [])))} | "
        f"medium_tags: {ascii_text(short_tags(item.get('medium_tags', [])))}"
    )

    y = PADDING
    draw.text((PADDING, y), header_1, font=font, fill=TEXT)
    y += 16
    y = draw_wrapped_text(draw, (PADDING, y), header_2, 128, font, MUTED)
    draw_wrapped_text(draw, (PADDING, y), header_3, 128, font, MUTED)

    for idx, variant_name in enumerate(VARIANT_ORDER):
        x = PADDING + idx * (CELL_SIZE + PADDING)
        image_y = HEADER_HEIGHT
        caption_y = HEADER_HEIGHT + CELL_SIZE + 8

        if variant_name == "target":
            source_path = variant_rows["baseline"]["target_path"]
            lines = ["reference image", Path(source_path).name[:40]]
        else:
            source_path = variant_rows[variant_name]["after_path"]
            lines = metric_line(variant_rows[variant_name])

        image = open_image(source_path).resize((CELL_SIZE, CELL_SIZE))
        canvas.paste(image, (x, image_y))
        draw.rectangle(
            (x - 1, image_y - 1, x + CELL_SIZE, image_y + CELL_SIZE),
            outline=BORDER,
            width=1,
        )
        draw.text((x, caption_y), ascii_text(VARIANT_DISPLAY[variant_name]), font=font, fill=TEXT)
        line_y = caption_y + 14
        for line in lines:
            draw.text((x, line_y), ascii_text(line[:44]), font=font, fill=MUTED)
            line_y += 12

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_packet_items(
    human_review_queue: List[Dict[str, Any]],
    style_summary: List[Dict[str, Any]],
    sample_diagnostics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sample_by_request_id = {row["request_id"]: row for row in sample_diagnostics}
    selected_request_ids = set()
    items: List[Dict[str, Any]] = []

    for row in human_review_queue:
        sample = sample_by_request_id[row["request_id"]]
        item = {
            "section": "safe_gain_queue",
            "request_id": row["request_id"],
            "style_name": row["style_name"],
            "best_variant_vs_baseline": row["best_variant_vs_baseline"],
            "reason": (
                f"safe gain over baseline: sem={row['semantic_wins']} tex={row['texture_wins']}"
            ),
            "texture_tags": sample.get("texture_tags", []),
            "medium_tags": sample.get("medium_tags", []),
        }
        items.append(item)
        selected_request_ids.add(item["request_id"])

    zero_safe_styles = []
    heavy_prompt_styles = []
    for row in style_summary:
        total_safe = sum(
            row["vs_baseline"][variant]["safe_gain"]
            for variant in ["prob035", "prob015", "brushstroke"]
        )
        if total_safe == 0:
            zero_safe_styles.append(row)
        if row["heavy_prompt_semantic_signals"] > row["light_prompt_semantic_signals"]:
            heavy_prompt_styles.append(row)

    for row in sorted(
        zero_safe_styles,
        key=lambda item: (
            item["heavy_prompt_semantic_signals"],
            item["count"],
            item["group_name"],
        ),
        reverse=True,
    ):
        for request_id in row["request_ids"]:
            if request_id in selected_request_ids:
                continue
            sample = sample_by_request_id[request_id]
            item = {
                "section": "no_safe_gain_watchlist",
                "request_id": request_id,
                "style_name": row["group_name"],
                "best_variant_vs_baseline": "-",
                "reason": "no texture branch reaches safe_gain for this style",
                "texture_tags": sample.get("texture_tags", []),
                "medium_tags": sample.get("medium_tags", []),
            }
            items.append(item)
            selected_request_ids.add(request_id)

    for row in sorted(
        heavy_prompt_styles,
        key=lambda item: (
            item["heavy_prompt_semantic_signals"] - item["light_prompt_semantic_signals"],
            item["count"],
            item["group_name"],
        ),
        reverse=True,
    ):
        for request_id in row["request_ids"]:
            if request_id in selected_request_ids:
                continue
            sample = sample_by_request_id[request_id]
            item = {
                "section": "heavy_prompt_risk",
                "request_id": request_id,
                "style_name": row["group_name"],
                "best_variant_vs_baseline": "-",
                "reason": "style-level semantic signal leans to heavier or broader prompt",
                "texture_tags": sample.get("texture_tags", []),
                "medium_tags": sample.get("medium_tags", []),
            }
            items.append(item)
            selected_request_ids.add(request_id)

    return items


def group_items(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["section"], []).append(item)
    return groups


def relative_link(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def build_markdown(
    items: List[Dict[str, Any]],
    variant_rows: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path,
) -> str:
    groups = group_items(items)
    lines: List[str] = []
    lines.append("# EmoArt Texture Human Review Packet")
    lines.append("")
    lines.append("## How To Use")
    lines.append("- Each contact sheet shows: `target | baseline | prob0.35 | prob0.15 | prob0.15+brushstroke_only`.")
    lines.append("- Under each generated image: first line is `nll / psnr / mae`, second line is `gradient / laplacian / high-frequency abs error`.")
    lines.append("- `safe_gain_queue` is the highest-value human review set because the variant beats baseline on both semantic and texture majorities.")
    lines.append("- `no_safe_gain_watchlist` marks styles where none of the texture branches became a stable replacement on the aligned 32-sample study.")
    lines.append("- `heavy_prompt_risk` marks remaining samples from styles whose semantic signal leans toward heavier or broader prompt injection.")
    lines.append("")

    section_titles = {
        "safe_gain_queue": "Safe Gain Queue",
        "no_safe_gain_watchlist": "No-Safe-Gain Watchlist",
        "heavy_prompt_risk": "Heavy Prompt Risk",
    }

    for section_name in ["safe_gain_queue", "no_safe_gain_watchlist", "heavy_prompt_risk"]:
        section_items = groups.get(section_name, [])
        if not section_items:
            continue
        lines.append(f"## {section_titles[section_name]}")
        lines.append("| request_id | style | best variant | reason | sheet |")
        lines.append("|---|---|---|---|---|")
        for item in section_items:
            sheet_name = f"{item['request_id'].replace('/', '_')}.png"
            sheet_path = output_dir / "sheets" / sheet_name
            lines.append(
                f"| {item['request_id']} | {item['style_name']} | {item['best_variant_vs_baseline']} | "
                f"{item['reason']} | [{sheet_name}]({relative_link(sheet_path, output_dir)}) |"
            )
        lines.append("")

        for item in section_items:
            request_id = item["request_id"]
            sheet_name = f"{request_id.replace('/', '_')}.png"
            sheet_path = output_dir / "sheets" / sheet_name
            baseline_row = variant_rows["baseline"][request_id]
            lines.append(f"### {request_id}")
            lines.append(f"- style: `{item['style_name']}`")
            lines.append(f"- reason: `{item['reason']}`")
            lines.append(f"- texture_tags: `{short_tags(item.get('texture_tags', []), max_items=8)}`")
            lines.append(f"- medium_tags: `{short_tags(item.get('medium_tags', []), max_items=8)}`")
            lines.append(f"- sheet: `{sheet_path}`")
            lines.append(f"- target image: `{baseline_row['image_path']}`")
            lines.append(f"- baseline image: `{variant_rows['baseline'][request_id]['after_path']}`")
            lines.append(f"- prob035 image: `{variant_rows['prob035'][request_id]['after_path']}`")
            lines.append(f"- prob015 image: `{variant_rows['prob015'][request_id]['after_path']}`")
            lines.append(f"- brushstroke image: `{variant_rows['brushstroke'][request_id]['after_path']}`")
            lines.append("")

    return "\n".join(lines) + "\n"


def build_manual_review_sheet(items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Texture Policy Manual Review Sheet")
    lines.append("")
    lines.append(
        "| section | request_id | style_name | candidate_variant | semantic_winner | texture_winner | fake_detail | accept_texture_branch | manual_note |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for item in items:
        lines.append(
            f"| {item['section']} | {item['request_id']} | {item['style_name']} | "
            f"{item['best_variant_vs_baseline']} |  |  | unsure | pending |  |"
        )
    lines.append("")
    lines.append("- `candidate_variant` is the model-side recommendation from the automated packet, not a final human verdict.")
    lines.append("- `accept_texture_branch` is intended for a binary call on whether this sample supports further style-aware policy exploration.")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build human-review packet for EmoArt texture policy study.")
    parser.add_argument(
        "--policy-review-dir",
        default="/root/autodl-tmp/emoart_gen_runs/texture_policy_review_32",
    )
    parser.add_argument(
        "--baseline-comparison",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_baseline_texture_32/comparison.jsonl",
    )
    parser.add_argument(
        "--prob035-comparison",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_texture_balanced_headonly_32/comparison.jsonl",
    )
    parser.add_argument(
        "--prob015-comparison",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_texture_balanced_headonly_prob015_32/comparison.jsonl",
    )
    parser.add_argument(
        "--brushstroke-comparison",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_texture_balanced_headonly_prob015_brushstroke_32/comparison.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/emoart_gen_runs/texture_policy_review_32/review_packet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy_review_dir = Path(args.policy_review_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    human_review_queue = load_json(policy_review_dir / "human_review_queue.json")
    style_summary = load_json(policy_review_dir / "style_summary.json")
    sample_diagnostics = load_json(policy_review_dir / "sample_diagnostics.json")

    comparison_rows = {
        "baseline": load_jsonl(Path(args.baseline_comparison)),
        "prob035": load_jsonl(Path(args.prob035_comparison)),
        "prob015": load_jsonl(Path(args.prob015_comparison)),
        "brushstroke": load_jsonl(Path(args.brushstroke_comparison)),
    }
    variant_rows = {
        variant_name: {row["request_id"]: row for row in rows}
        for variant_name, rows in comparison_rows.items()
    }

    items = build_packet_items(human_review_queue, style_summary, sample_diagnostics)
    for item in items:
        request_id = item["request_id"]
        sheet_name = f"{request_id.replace('/', '_')}.png"
        make_sheet(
            item=item,
            variant_rows={name: rows[request_id] for name, rows in variant_rows.items()},
            output_path=output_dir / "sheets" / sheet_name,
        )

    packet_markdown = build_markdown(items, variant_rows, output_dir)
    (output_dir / "review_packet.md").write_text(packet_markdown, encoding="utf-8")
    (output_dir / "manual_review_sheet.md").write_text(
        build_manual_review_sheet(items),
        encoding="utf-8",
    )
    (output_dir / "packet_manifest.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
