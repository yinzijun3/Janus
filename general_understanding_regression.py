from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL_PATH = os.environ.get(
    "JANUS_MODEL_PATH",
    "/root/autodl-tmp/hf_cache/hub/models--deepseek-ai--Janus-Pro-1B/snapshots/960ab33191f61342a4c60ae74d8dc356a39fafcb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="General image-understanding regression for Janus base vs adapter.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_dtype(dtype_name: str) -> torch.dtype:
    import torch

    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(model_path: str, dtype: torch.dtype, adapter_path: Optional[str] = None):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from janus.models import MultiModalityCausalLM, VLChatProcessor

    processor = VLChatProcessor.from_pretrained(model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if adapter_path:
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            adapter_path,
            is_trainable=False,
        )
    model = model.to(dtype).cuda().eval()
    return model, processor


def generate_answer(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch
    from PIL import Image

    with torch.inference_mode():
        with Image.open(image_path) as image:
            pil_image = image.convert("RGB")
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepared = processor(
            conversations=conversation,
            images=[pil_image],
            force_batchify=True,
        ).to(model.device)
        inputs_embeds = model.prepare_inputs_embeds(**prepared)
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        return processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def contains_keyword(text: str, keyword: str) -> bool:
    text = normalize_text(text)
    keyword = normalize_text(keyword)
    if not keyword:
        return False
    return keyword in text


def keyword_coverage(text: str, keywords: List[str]) -> Optional[float]:
    if not keywords:
        return None
    hits = sum(1 for keyword in keywords if contains_keyword(text, keyword))
    return hits / max(len(keywords), 1)


def forbidden_absence_rate(text: str, forbidden_keywords: List[str]) -> Optional[float]:
    if not forbidden_keywords:
        return None
    misses = sum(1 for keyword in forbidden_keywords if not contains_keyword(text, keyword))
    return misses / max(len(forbidden_keywords), 1)


def maybe_compute_rouge(predictions: List[str], references: List[str]) -> Optional[float]:
    if not references:
        return None
    try:
        from rouge_score import rouge_scorer
    except Exception:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(reference, prediction)["rougeL"].fmeasure for prediction, reference in zip(predictions, references)]
    return sum(scores) / max(len(scores), 1)


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_keyword_scores = [row["base_keyword_coverage"] for row in rows if row["base_keyword_coverage"] is not None]
    tuned_keyword_scores = [row["adapter_keyword_coverage"] for row in rows if row["adapter_keyword_coverage"] is not None]
    base_forbidden_scores = [row["base_forbidden_absence"] for row in rows if row["base_forbidden_absence"] is not None]
    tuned_forbidden_scores = [row["adapter_forbidden_absence"] for row in rows if row["adapter_forbidden_absence"] is not None]
    references = [row["reference"] for row in rows if row.get("reference")]
    base_predictions = [row["base_answer"] for row in rows if row.get("reference")]
    tuned_predictions = [row["adapter_answer"] for row in rows if row.get("reference")]
    return {
        "num_samples": len(rows),
        "num_reference_samples": len(references),
        "avg_base_keyword_coverage": sum(base_keyword_scores) / len(base_keyword_scores) if base_keyword_scores else None,
        "avg_adapter_keyword_coverage": sum(tuned_keyword_scores) / len(tuned_keyword_scores) if tuned_keyword_scores else None,
        "avg_base_forbidden_absence": sum(base_forbidden_scores) / len(base_forbidden_scores) if base_forbidden_scores else None,
        "avg_adapter_forbidden_absence": sum(tuned_forbidden_scores) / len(tuned_forbidden_scores) if tuned_forbidden_scores else None,
        "base_rougeL": maybe_compute_rouge(base_predictions, references),
        "adapter_rougeL": maybe_compute_rouge(tuned_predictions, references),
    }


def write_markdown(path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# General Understanding Regression")
    lines.append("")
    lines.append("## Summary")
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Samples")
    for row in rows:
        lines.append(f"### {row['id']}")
        lines.append(f"- category: `{row.get('category', '')}`")
        lines.append(f"- image_path: `{row['image_path']}`")
        lines.append(f"- prompt: `{row['prompt']}`")
        if row.get("review_focus"):
            lines.append(f"- review_focus: `{row['review_focus']}`")
        if row.get("reference"):
            lines.append(f"- reference: `{row['reference']}`")
        if row.get("expected_keywords"):
            lines.append(f"- expected_keywords: `{', '.join(row['expected_keywords'])}`")
        if row.get("forbidden_keywords"):
            lines.append(f"- forbidden_keywords: `{', '.join(row['forbidden_keywords'])}`")
        lines.append(f"- base_answer: `{row['base_answer']}`")
        lines.append(f"- adapter_answer: `{row['adapter_answer']}`")
        lines.append(
            f"- base_keyword_coverage / adapter_keyword_coverage: "
            f"{row['base_keyword_coverage']} / {row['adapter_keyword_coverage']}"
        )
        lines.append(
            f"- base_forbidden_absence / adapter_forbidden_absence: "
            f"{row['base_forbidden_absence']} / {row['adapter_forbidden_absence']}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    records = load_jsonl(args.manifest_path)

    dtype = get_dtype(args.dtype)
    base_model, base_processor = load_model(args.model_path, dtype)
    adapter_model, adapter_processor = load_model(args.model_path, dtype, adapter_path=args.adapter_path)

    rows: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        sample_id = record.get("id") or record.get("request_id") or f"sample_{index:03d}"
        image_path = record["image_path"]
        prompt = record["prompt"]
        reference = record.get("reference")
        expected_keywords = record.get("expected_keywords", [])
        forbidden_keywords = record.get("forbidden_keywords", [])

        base_answer = generate_answer(base_model, base_processor, image_path, prompt, args.max_new_tokens)
        adapter_answer = generate_answer(adapter_model, adapter_processor, image_path, prompt, args.max_new_tokens)

        rows.append(
            {
                "id": sample_id,
                "category": record.get("category"),
                "image_path": image_path,
                "prompt": prompt,
                "review_focus": record.get("review_focus"),
                "reference": reference,
                "expected_keywords": expected_keywords,
                "forbidden_keywords": forbidden_keywords,
                "base_answer": base_answer,
                "adapter_answer": adapter_answer,
                "base_keyword_coverage": keyword_coverage(base_answer, expected_keywords),
                "adapter_keyword_coverage": keyword_coverage(adapter_answer, expected_keywords),
                "base_forbidden_absence": forbidden_absence_rate(base_answer, forbidden_keywords),
                "adapter_forbidden_absence": forbidden_absence_rate(adapter_answer, forbidden_keywords),
            }
        )

    summary = summarize_results(rows)
    summary["model_path"] = args.model_path
    summary["adapter_path"] = args.adapter_path
    summary["manifest_path"] = args.manifest_path

    output_dir = Path(args.output_dir)
    (output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_dir / "report.md", summary, rows)


if __name__ == "__main__":
    main()
