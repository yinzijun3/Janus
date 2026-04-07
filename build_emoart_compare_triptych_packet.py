import argparse
import json
import textwrap
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


COLUMNS = [
    ("target", "target"),
    ("before", "raw"),
    ("after", "adapter"),
]

CELL_SIZE = 256
PADDING = 16
HEADER_HEIGHT = 138
CAPTION_HEIGHT = 72
BACKGROUND = (248, 246, 241)
TEXT = (20, 20, 20)
MUTED = (90, 90, 90)
BORDER = (210, 205, 198)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ascii_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return normalized.encode("ascii", "ignore").decode("ascii")


def short_text(value: str, width: int = 120) -> str:
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


def before_metric_lines(row: Dict[str, Any]) -> List[str]:
    lines = [
        (
            f"nll {row['reference_token_nll_before']:.3f}  "
            f"psnr {row['reference_psnr_before']:.2f}  "
            f"mae {row['reference_mae_before']:.1f}"
        ),
    ]
    if "reference_gradient_energy_abs_error_before" in row:
        lines.append(
            (
                f"g {row['reference_gradient_energy_abs_error_before']:.1f}  "
                f"l {row['reference_laplacian_variance_abs_error_before']:.0f}  "
                f"h {row['reference_high_frequency_energy_abs_error_before']:.0f}"
            )
        )
    return lines


def after_metric_lines(row: Dict[str, Any]) -> List[str]:
    delta_psnr = row["reference_psnr_after"] - row["reference_psnr_before"]
    delta_mae = row["reference_mae_after"] - row["reference_mae_before"]
    delta_nll = row["reference_token_nll_after"] - row["reference_token_nll_before"]

    lines = [
        (
            f"nll {row['reference_token_nll_after']:.3f}  "
            f"psnr {row['reference_psnr_after']:.2f}  "
            f"mae {row['reference_mae_after']:.1f}"
        ),
        f"d nll {delta_nll:+.2f}  d psnr {delta_psnr:+.2f}  d mae {delta_mae:+.1f}",
    ]
    if "reference_gradient_energy_abs_error_after" in row:
        lines.append(
            (
                f"g {row['reference_gradient_energy_abs_error_after']:.1f}  "
                f"l {row['reference_laplacian_variance_abs_error_after']:.0f}  "
                f"h {row['reference_high_frequency_energy_abs_error_after']:.0f}"
            )
        )
    return lines


def score_row(row: Dict[str, Any]) -> float:
    return (
        (row["reference_psnr_after"] - row["reference_psnr_before"])
        + (row["reference_token_nll_before"] - row["reference_token_nll_after"])
        + (row["reference_mae_before"] - row["reference_mae_after"]) / 10.0
    )


def build_shortlist(rows: List[Dict[str, Any]], n_gain: int = 4, n_risk: int = 4) -> List[Dict[str, Any]]:
    gain_sorted = sorted(rows, key=score_row, reverse=True)
    risk_sorted = sorted(rows, key=score_row)
    selected: List[Dict[str, Any]] = []
    seen = set()
    for row in gain_sorted[:n_gain] + risk_sorted[:n_risk]:
        if row["request_id"] in seen:
            continue
        selected.append(row)
        seen.add(row["request_id"])
    return selected


def make_sheet(row: Dict[str, Any], output_path: Path) -> None:
    width = PADDING * 2 + len(COLUMNS) * CELL_SIZE + (len(COLUMNS) - 1) * PADDING
    height = HEADER_HEIGHT + CELL_SIZE + CAPTION_HEIGHT + PADDING * 2
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    header_1 = f"{ascii_text(row['request_id'])} | idx={row['index']:02d}"
    header_2 = f"target: {ascii_text(Path(row['image_path']).name)}"
    header_3 = f"prompt: {ascii_text(short_text(row['prompt'], 150))}"

    y = PADDING
    draw.text((PADDING, y), header_1, font=font, fill=TEXT)
    y += 16
    draw.text((PADDING, y), header_2, font=font, fill=MUTED)
    y += 14
    draw_wrapped_text(draw, (PADDING, y), header_3, 116, font, MUTED)

    for idx, (column_name, label) in enumerate(COLUMNS):
        x = PADDING + idx * (CELL_SIZE + PADDING)
        image_y = HEADER_HEIGHT
        caption_y = HEADER_HEIGHT + CELL_SIZE + 8

        if column_name == "target":
            source_path = row["target_path"]
            lines = ["reference image", Path(source_path).name]
        elif column_name == "before":
            source_path = row["before_path"]
            lines = before_metric_lines(row)
        else:
            source_path = row["after_path"]
            lines = after_metric_lines(row)

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
            draw.text((x, line_y), ascii_text(line[:46]), font=font, fill=MUTED)
            line_y += 12

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_review_packet(rows: List[Dict[str, Any]], output_dir: Path) -> str:
    shortlist = build_shortlist(rows)
    lines: List[str] = []
    lines.append("# EmoArt Compare Triptych Packet")
    lines.append("")
    lines.append("## How To Read")
    lines.append("- Each sheet shows: `target | raw | adapter`.")
    lines.append("- `raw` is the original Janus-Pro output under the same prompt.")
    lines.append("- `adapter` is the finetuned model output under the same prompt.")
    lines.append("- Under `adapter`, the second line is the delta vs `raw` for `nll / psnr / mae`.")
    lines.append("")
    lines.append("## Shortlist")
    lines.append("| request_id | idx | reason | sheet |")
    lines.append("|---|---:|---|---|")
    shortlist_ids = {row["request_id"] for row in shortlist}
    for row in shortlist:
        reason = "gain" if score_row(row) >= 0 else "risk"
        sheet_name = f"{row['request_id'].replace('/', '_')}.png"
        lines.append(
            f"| {row['request_id']} | {row['index']:02d} | {reason} | "
            f"[{sheet_name}](sheets/{sheet_name}) |"
        )
    lines.append("")
    lines.append("## All Samples")
    lines.append("| idx | request_id | improved_vs_reference | sheet |")
    lines.append("|---:|---|---|---|")
    for row in rows:
        sheet_name = f"{row['request_id'].replace('/', '_')}.png"
        lines.append(
            f"| {row['index']:02d} | {row['request_id']} | {row.get('improved_vs_reference', '')} | "
            f"[{sheet_name}](sheets/{sheet_name}) |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_manual_review_sheet(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# EmoArt Compare Triptych Manual Review Sheet")
    lines.append("")
    lines.append("| idx | request_id | baseline_vs_raw | line_complexity | border_bias | style_bias | note |")
    lines.append("|---:|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['index']:02d} | {row['request_id']} | pending | pending | pending | pending |  |"
        )
    lines.append("")
    lines.append("- `baseline_vs_raw`: `better / worse / tie`")
    lines.append("- `line_complexity`: did the adapter preserve or improve complex line/detail behavior?")
    lines.append("- `border_bias`: `none / mild / clear`")
    lines.append("- `style_bias`: note any strong regional or style leakage such as ink-wash or scroll/frame bias")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a triptych review packet from compare_emoart_gen output.")
    parser.add_argument("--compare-dir", required=True, help="Directory containing comparison.jsonl and before/after/target.")
    parser.add_argument("--output-dir", default=None, help="Output directory for the review packet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_dir = Path(args.compare_dir)
    output_dir = Path(args.output_dir) if args.output_dir else compare_dir / "triptych_packet"
    comparison_path = compare_dir / "comparison.jsonl"
    if not comparison_path.exists():
        raise SystemExit(
            "comparison.jsonl not found in compare dir: "
            f"{comparison_path}\n"
            "Run compare_emoart_gen.py first, then build the triptych packet."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(comparison_path)
    rows = sorted(rows, key=lambda row: row["index"])

    for row in rows:
        sheet_name = f"{row['request_id'].replace('/', '_')}.png"
        make_sheet(row, output_dir / "sheets" / sheet_name)

    (output_dir / "review_packet.md").write_text(build_review_packet(rows, output_dir), encoding="utf-8")
    (output_dir / "manual_review_sheet.md").write_text(build_manual_review_sheet(rows), encoding="utf-8")
    (output_dir / "packet_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
