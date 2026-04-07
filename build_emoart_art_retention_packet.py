import argparse
import json
import textwrap
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


COLUMNS = [
    ("target", "target"),
    ("raw_base", "raw"),
    ("baseline", "baseline"),
    ("raw_prob035", "raw"),
    ("prob035", "probe 0.35"),
    ("raw_prob015", "raw"),
    ("prob015", "probe 0.15"),
    ("raw_brushstroke", "raw"),
    ("brushstroke", "brush"),
]

CELL_SIZE = 170
PADDING = 10
HEADER_HEIGHT = 150
CAPTION_HEIGHT = 60
BACKGROUND = (248, 246, 241)
TEXT = (20, 20, 20)
MUTED = (90, 90, 90)
BORDER = (210, 205, 198)

SHORTLIST_IDS = [
    "Korea_request-81",
    "Realism_request-69",
    "Post-Impressionism_request-20",
    "Orientalism_request-43",
    "Dada_request-87",
    "High Renaissance_request-69",
    "Baroque_request-4",
    "Regionalism_request-24",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ascii_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii")


def short_text(value: str, width: int = 112) -> str:
    value = " ".join((value or "").split())
    if len(value) <= width:
        return value
    return value[: width - 3] + "..."


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


def before_metric_line(row: Dict[str, Any]) -> List[str]:
    return [
        (
            f"nll {row['reference_token_nll_before']:.3f}  "
            f"psnr {row['reference_psnr_before']:.2f}  "
            f"mae {row['reference_mae_before']:.1f}"
        ),
        (
            f"g {row['reference_gradient_energy_abs_error_before']:.1f}  "
            f"l {row['reference_laplacian_variance_abs_error_before']:.0f}  "
            f"h {row['reference_high_frequency_energy_abs_error_before']:.0f}"
        ),
    ]


def after_metric_line(row: Dict[str, Any]) -> List[str]:
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


def delta_metric_line(row: Dict[str, Any]) -> str:
    delta_psnr = row["reference_psnr_after"] - row["reference_psnr_before"]
    delta_mae = row["reference_mae_after"] - row["reference_mae_before"]
    delta_nll = row["reference_token_nll_after"] - row["reference_token_nll_before"]
    return f"d nll {delta_nll:+.2f}  d psnr {delta_psnr:+.2f}  d mae {delta_mae:+.1f}"


def short_tags(tags: List[str], max_items: int = 4) -> str:
    if not tags:
        return "-"
    if len(tags) <= max_items:
        return ", ".join(tags)
    return ", ".join(tags[:max_items]) + f" (+{len(tags) - max_items})"


def build_item_list(
    baseline_rows: List[Dict[str, Any]],
    sample_diagnostics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    diagnostics_by_request_id = {row["request_id"]: row for row in sample_diagnostics}
    items = []
    for row in baseline_rows:
        diagnostic = diagnostics_by_request_id.get(row["request_id"], {})
        items.append(
            {
                "index": row["index"],
                "request_id": row["request_id"],
                "style_name": diagnostic.get("style_name", row["request_id"].split("_request-")[0]),
                "prompt": row["prompt"],
                "image_path": row["image_path"],
                "texture_tags": diagnostic.get("texture_tags", []),
                "medium_tags": diagnostic.get("medium_tags", []),
            }
        )
    return items


def make_sheet(
    item: Dict[str, Any],
    row_bundle: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    width = PADDING * 2 + len(COLUMNS) * CELL_SIZE + (len(COLUMNS) - 1) * PADDING
    height = HEADER_HEIGHT + CELL_SIZE + CAPTION_HEIGHT + PADDING * 2
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    header_1 = f"{ascii_text(item['request_id'])} | {ascii_text(item['style_name'])} | idx={item['index']:02d}"
    header_2 = f"target: {ascii_text(Path(item['image_path']).name)}"
    header_3 = f"prompt: {ascii_text(short_text(item['prompt'], 150))}"
    header_4 = (
        f"texture_tags: {ascii_text(short_tags(item.get('texture_tags', []), 6))} | "
        f"medium_tags: {ascii_text(short_tags(item.get('medium_tags', []), 6))}"
    )

    y = PADDING
    draw.text((PADDING, y), header_1, font=font, fill=TEXT)
    y += 16
    draw.text((PADDING, y), header_2, font=font, fill=MUTED)
    y += 14
    y = draw_wrapped_text(draw, (PADDING, y), header_3, 180, font, MUTED)
    draw_wrapped_text(draw, (PADDING, y), header_4, 180, font, MUTED)

    for idx, (column_name, label) in enumerate(COLUMNS):
        x = PADDING + idx * (CELL_SIZE + PADDING)
        image_y = HEADER_HEIGHT
        caption_y = HEADER_HEIGHT + CELL_SIZE + 8

        if column_name == "target":
            source_path = row_bundle["baseline"]["target_path"]
            lines = ["reference", Path(source_path).name]
        elif column_name == "raw_base":
            source_path = row_bundle["baseline"]["before_path"]
            lines = before_metric_line(row_bundle["baseline"])
        elif column_name == "baseline":
            source_path = row_bundle["baseline"]["after_path"]
            lines = after_metric_line(row_bundle["baseline"]) + [delta_metric_line(row_bundle["baseline"])]
        elif column_name == "raw_prob035":
            source_path = row_bundle["prob035"]["before_path"]
            lines = before_metric_line(row_bundle["prob035"])
        elif column_name == "prob035":
            source_path = row_bundle["prob035"]["after_path"]
            lines = after_metric_line(row_bundle["prob035"]) + [delta_metric_line(row_bundle["prob035"])]
        elif column_name == "raw_prob015":
            source_path = row_bundle["prob015"]["before_path"]
            lines = before_metric_line(row_bundle["prob015"])
        elif column_name == "prob015":
            source_path = row_bundle["prob015"]["after_path"]
            lines = after_metric_line(row_bundle["prob015"]) + [delta_metric_line(row_bundle["prob015"])]
        elif column_name == "raw_brushstroke":
            source_path = row_bundle["brushstroke"]["before_path"]
            lines = before_metric_line(row_bundle["brushstroke"])
        else:
            source_path = row_bundle["brushstroke"]["after_path"]
            lines = after_metric_line(row_bundle["brushstroke"]) + [delta_metric_line(row_bundle["brushstroke"])]

        image = open_image(source_path).resize((CELL_SIZE, CELL_SIZE))
        canvas.paste(image, (x, image_y))
        draw.rectangle(
            (x - 1, image_y - 1, x + CELL_SIZE, image_y + CELL_SIZE),
            outline=BORDER,
            width=1,
        )
        draw.text((x, caption_y), ascii_text(label), font=font, fill=TEXT)
        line_y = caption_y + 14
        for line in lines[:3]:
            draw.text((x, line_y), ascii_text(line[:32]), font=font, fill=MUTED)
            line_y += 11

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_markdown(items: List[Dict[str, Any]], output_dir: Path) -> str:
    shortlist = [item for item in items if item["request_id"] in SHORTLIST_IDS]
    all_items = sorted(items, key=lambda item: item["index"])
    lines: List[str] = []
    lines.append("# EmoArt Art Generation Retention Visual Packet")
    lines.append("")
    lines.append("## How To Read")
    lines.append("- Each sheet shows: `target | raw(base prompt) | baseline | raw(prob0.35 prompt) | prob0.35 | raw(prob0.15 prompt) | prob0.15 | raw(brushstroke prompt) | brushstroke`.")
    lines.append("- `raw(...)` means original `Janus-Pro-1B` under that exact prompt variant.")
    lines.append("- `baseline / probe` columns show the corresponding adapter result under the same prompt variant.")
    lines.append("- Under each adapted column, the third metric line is the delta vs its paired raw result.")
    lines.append("")

    lines.append("## Representative Shortlist")
    lines.append("| request_id | style | why look first | sheet |")
    lines.append("|---|---|---|---|")
    shortlist_notes = {
        "Korea_request-81": "strong task retention and large gain over raw",
        "Realism_request-69": "clean example where texture branch is worth checking",
        "Post-Impressionism_request-20": "brushstroke-style opt-in candidate",
        "Orientalism_request-43": "shows texture/semantic tradeoff behavior clearly",
        "Dada_request-87": "strong brushstroke-side gain candidate",
        "High Renaissance_request-69": "baseline is strong while probes vary",
        "Baroque_request-4": "clear raw-to-adapter improvement on art portrait prompt",
        "Regionalism_request-24": "important failure case where not every sample improves",
    }
    for item in shortlist:
        sheet_name = f"{item['request_id'].replace('/', '_')}.png"
        lines.append(
            f"| {item['request_id']} | {item['style_name']} | {shortlist_notes.get(item['request_id'], '-')} | "
            f"[{sheet_name}](sheets/{sheet_name}) |"
        )
    lines.append("")

    lines.append("## All 32 Samples")
    lines.append("| idx | request_id | style | sheet |")
    lines.append("|---:|---|---|---|")
    for item in all_items:
        sheet_name = f"{item['request_id'].replace('/', '_')}.png"
        lines.append(
            f"| {item['index']:02d} | {item['request_id']} | {item['style_name']} | "
            f"[{sheet_name}](sheets/{sheet_name}) |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_manual_review_sheet(items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# EmoArt Art Generation Retention Manual Review Sheet")
    lines.append("")
    lines.append("| idx | request_id | style | baseline_vs_raw | best_probe | probe_beats_baseline | note |")
    lines.append("|---:|---|---|---|---|---|---|")
    for item in sorted(items, key=lambda row: row["index"]):
        lines.append(
            f"| {item['index']:02d} | {item['request_id']} | {item['style_name']} | pending | pending | pending |  |"
        )
    lines.append("")
    lines.append("- `baseline_vs_raw`: does Exp A baseline clearly preserve or improve art-task generation relative to original Janus-Pro?")
    lines.append("- `best_probe`: choose from `baseline / prob035 / prob015 / brushstroke`.")
    lines.append("- `probe_beats_baseline`: `yes / no / local-only`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual packet for EmoArt art-generation retention study.")
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
        "--sample-diagnostics",
        default="/root/autodl-tmp/emoart_gen_runs/texture_policy_review_32/sample_diagnostics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/emoart_gen_runs/art_retention_visual_packet_32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_variant = {
        "baseline": {row["request_id"]: row for row in load_jsonl(Path(args.baseline_comparison))},
        "prob035": {row["request_id"]: row for row in load_jsonl(Path(args.prob035_comparison))},
        "prob015": {row["request_id"]: row for row in load_jsonl(Path(args.prob015_comparison))},
        "brushstroke": {row["request_id"]: row for row in load_jsonl(Path(args.brushstroke_comparison))},
    }
    baseline_rows = list(rows_by_variant["baseline"].values())
    sample_diagnostics = load_json(Path(args.sample_diagnostics))
    items = build_item_list(baseline_rows, sample_diagnostics)

    for item in items:
        request_id = item["request_id"]
        row_bundle = {name: rows[request_id] for name, rows in rows_by_variant.items()}
        sheet_name = f"{request_id.replace('/', '_')}.png"
        make_sheet(item, row_bundle, output_dir / "sheets" / sheet_name)

    (output_dir / "review_packet.md").write_text(build_markdown(items, output_dir), encoding="utf-8")
    (output_dir / "manual_review_sheet.md").write_text(build_manual_review_sheet(items), encoding="utf-8")
    (output_dir / "packet_manifest.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
