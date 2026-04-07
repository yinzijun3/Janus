from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKGROUND = (247, 245, 240)
TEXT = (22, 22, 22)
MUTED = (92, 92, 92)
BORDER = (208, 202, 194)
PADDING = 16
CELL_SIZE = 384
HEADER_HEIGHT = 118
CAPTION_HEIGHT = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="General image-generation regression for Janus base vs adapter.")
    parser.add_argument("--model-path", default=os.environ.get("JANUS_MODEL_PATH", "/root/autodl-tmp/hf_cache/hub/models--deepseek-ai--Janus-Pro-1B/snapshots/960ab33191f61342a4c60ae74d8dc356a39fafcb"))
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--cfg-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--parallel-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-token-num-per-image", type=int, default=576)
    parser.add_argument("--sample-strategy", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--use-clipscore", action="store_true")
    parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch32")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width_chars: int, font, fill) -> int:
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars) or [text]:
        draw.text((x, y), line, font=font, fill=fill)
        y += 13
    return y


def write_sheet(
    sample_id: str,
    category: str,
    prompt: str,
    base_image_path: Path,
    adapter_image_path: Path,
    output_path: Path,
    base_clipscore: Optional[float],
    adapter_clipscore: Optional[float],
) -> None:
    import PIL.Image
    from PIL import Image, ImageDraw, ImageFont

    width = PADDING * 3 + CELL_SIZE * 2
    height = PADDING * 2 + HEADER_HEIGHT + CELL_SIZE + CAPTION_HEIGHT
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    y = PADDING
    draw.text((PADDING, y), sample_id, font=font, fill=TEXT)
    y += 16
    if category:
        draw.text((PADDING, y), f"category={category}", font=font, fill=MUTED)
        y += 14
    draw_wrapped_text(draw, (PADDING, y), prompt, 120, font, MUTED)

    base_image = PIL.Image.open(base_image_path).convert("RGB").resize((CELL_SIZE, CELL_SIZE))
    adapter_image = PIL.Image.open(adapter_image_path).convert("RGB").resize((CELL_SIZE, CELL_SIZE))

    base_x = PADDING
    adapter_x = PADDING * 2 + CELL_SIZE
    image_y = PADDING + HEADER_HEIGHT
    caption_y = image_y + CELL_SIZE + 8

    canvas.paste(base_image, (base_x, image_y))
    canvas.paste(adapter_image, (adapter_x, image_y))
    draw.rectangle((base_x - 1, image_y - 1, base_x + CELL_SIZE, image_y + CELL_SIZE), outline=BORDER, width=1)
    draw.rectangle((adapter_x - 1, image_y - 1, adapter_x + CELL_SIZE, image_y + CELL_SIZE), outline=BORDER, width=1)

    draw.text((base_x, caption_y), "base", font=font, fill=TEXT)
    draw.text((adapter_x, caption_y), "adapter", font=font, fill=TEXT)
    if base_clipscore is not None:
        draw.text((base_x, caption_y + 14), f"clip {base_clipscore:.4f}", font=font, fill=MUTED)
    if adapter_clipscore is not None:
        draw.text((adapter_x, caption_y + 14), f"clip {adapter_clipscore:.4f}", font=font, fill=MUTED)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_markdown(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# General Generation Regression")
    lines.append("")
    lines.append("## Summary")
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Samples")
    lines.append("| id | category | review_focus | sheet | base_image | adapter_image |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['id']} | {row.get('category', '')} | {row.get('review_focus', '')} | "
            f"[sheet]({row['sheet_relpath']}) | [base]({row['base_relpath']}) | [adapter]({row['adapter_relpath']}) |"
        )
    lines.append("")
    for row in rows:
        lines.append(f"### {row['id']}")
        lines.append(f"- category: `{row.get('category', '')}`")
        lines.append(f"- prompt: `{row['prompt']}`")
        if row.get("review_focus"):
            lines.append(f"- review_focus: `{row['review_focus']}`")
        lines.append(f"- base_image: `{row['base_image_path']}`")
        lines.append(f"- adapter_image: `{row['adapter_image_path']}`")
        lines.append(f"- sheet: `{row['sheet_path']}`")
        if row.get("base_clipscore") is not None:
            lines.append(f"- base_clipscore / adapter_clipscore: `{row['base_clipscore']}` / `{row['adapter_clipscore']}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    from compare_emoart_gen import (
        apply_adapter,
        compute_clipscore,
        generate_images,
        get_dtype,
        load_model,
        maybe_load_clipscore,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)
    base_dir = output_dir / "base"
    adapter_dir = output_dir / "adapter"
    sheet_dir = output_dir / "sheets"
    base_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.manifest_path)
    dtype = get_dtype(args.dtype)
    base_model, processor = load_model(args.model_path, dtype)
    adapter_model, _ = load_model(args.model_path, dtype)
    adapter_model = apply_adapter(adapter_model, args.adapter_path)
    clip_bundle = maybe_load_clipscore(args.clip_model_name) if args.use_clipscore else None

    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt = record["prompt"]
        sample_id = record.get("id") or f"sample_{index:03d}"
        sample_seed = args.seed + index

        base_images, _ = generate_images(
            model=base_model,
            processor=processor,
            prompt=prompt,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            image_token_num_per_image=args.image_token_num_per_image,
            image_size=args.image_size,
            patch_size=args.patch_size,
            seed=sample_seed,
            sample_strategy=args.sample_strategy,
        )
        adapter_images, _ = generate_images(
            model=adapter_model,
            processor=processor,
            prompt=prompt,
            parallel_size=args.parallel_size,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            image_token_num_per_image=args.image_token_num_per_image,
            image_size=args.image_size,
            patch_size=args.patch_size,
            seed=sample_seed,
            sample_strategy=args.sample_strategy,
        )

        base_image = base_images[0]
        adapter_image = adapter_images[0]
        base_path = base_dir / f"{sample_id}.png"
        adapter_path = adapter_dir / f"{sample_id}.png"
        sheet_path = sheet_dir / f"{sample_id}.png"
        base_image.save(base_path)
        adapter_image.save(adapter_path)

        base_clipscore = compute_clipscore(clip_bundle, base_image, prompt)
        adapter_clipscore = compute_clipscore(clip_bundle, adapter_image, prompt)

        write_sheet(
            sample_id=sample_id,
            category=record.get("category", ""),
            prompt=prompt,
            base_image_path=base_path,
            adapter_image_path=adapter_path,
            output_path=sheet_path,
            base_clipscore=base_clipscore,
            adapter_clipscore=adapter_clipscore,
        )

        rows.append(
            {
                "id": sample_id,
                "category": record.get("category"),
                "prompt": prompt,
                "review_focus": record.get("review_focus"),
                "base_image_path": str(base_path),
                "adapter_image_path": str(adapter_path),
                "sheet_path": str(sheet_path),
                "base_relpath": base_path.relative_to(output_dir).as_posix(),
                "adapter_relpath": adapter_path.relative_to(output_dir).as_posix(),
                "sheet_relpath": sheet_path.relative_to(output_dir).as_posix(),
                "base_clipscore": base_clipscore,
                "adapter_clipscore": adapter_clipscore,
            }
        )

    summary: Dict[str, Any] = {
        "num_samples": len(rows),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "manifest_path": args.manifest_path,
    }
    clip_pairs = [(row["base_clipscore"], row["adapter_clipscore"]) for row in rows if row["base_clipscore"] is not None]
    if clip_pairs:
        summary["avg_base_clipscore"] = sum(pair[0] for pair in clip_pairs) / len(clip_pairs)
        summary["avg_adapter_clipscore"] = sum(pair[1] for pair in clip_pairs) / len(clip_pairs)

    (output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_dir / "report.md", summary, rows)
    build_manual_review_sheet(output_dir / "manual_review_sheet.md", rows)


def build_manual_review_sheet(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# General Generation Regression Manual Review Sheet")
    lines.append("")
    lines.append("| id | category | review_focus | semantic_control | style_pollution | base_better | adapter_better | tie | note |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['id']} | {row.get('category', '')} | {row.get('review_focus', '')} | pending | pending |  |  |  |  |"
        )
    lines.append("")
    lines.append("- `semantic_control`: does the adapter still follow the prompt cleanly?")
    lines.append("- `style_pollution`: does the adapter make the result look unnecessarily painterly or EmoArt-like?")
    lines.append("- mark exactly one of `base_better`, `adapter_better`, or `tie` per row.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
