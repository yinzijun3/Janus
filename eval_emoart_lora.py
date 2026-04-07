import argparse
import json
import os
import random

import torch
from peft import PeftModel
from PIL import Image
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM

from finetune.emoart import DEFAULT_PROMPT, load_jsonl, parse_json_response
from janus.models import MultiModalityCausalLM, VLChatProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base Janus and LoRA adapter on EmoArt-5k validation samples.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "JANUS_MODEL_PATH",
            "/root/autodl-tmp/hf_cache/hub/models--deepseek-ai--Janus-Pro-1B/snapshots/960ab33191f61342a4c60ae74d8dc356a39fafcb",
        ),
    )
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    return parser.parse_args()


def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(model_path, adapter_path=None):
    dtype = get_dtype()
    processor = VLChatProcessor.from_pretrained(model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if adapter_path:
        model.language_model = PeftModel.from_pretrained(model.language_model, adapter_path)
    model = model.to(dtype).cuda().eval()
    return model, processor, dtype


@torch.inference_mode()
def generate_prediction(model, processor, sample, max_new_tokens):
    with Image.open(sample["image_path"]) as image:
        pil_image = image.convert("RGB")
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{DEFAULT_PROMPT}",
            "images": [sample["image_path"]],
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


def compute_label_accuracy(predictions, references, key):
    total = 0
    correct = 0
    for prediction, reference in zip(predictions, references):
        parsed = parse_json_response(prediction)
        ref = reference.get(key)
        pred = parsed.get(key)
        if not ref:
            continue
        total += 1
        if pred == ref:
            correct += 1
    if total == 0:
        return None
    return correct / total


def compute_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(reference, prediction)["rougeL"].fmeasure for prediction, reference in zip(predictions, references)]
    return sum(scores) / max(len(scores), 1)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    records = load_jsonl(args.val_manifest)
    random.Random(args.seed).shuffle(records)
    samples = records[: args.num_samples]

    references = [json.dumps(sample["target_record"], ensure_ascii=False) for sample in samples]

    base_model, base_processor, _ = load_model(args.model_path)
    base_predictions = [
        generate_prediction(base_model, base_processor, sample, args.max_new_tokens) for sample in samples
    ]
    del base_model
    torch.cuda.empty_cache()

    tuned_model, tuned_processor, _ = load_model(args.model_path, adapter_path=args.adapter_path)
    tuned_predictions = [
        generate_prediction(tuned_model, tuned_processor, sample, args.max_new_tokens) for sample in samples
    ]

    summary = {
        "base_rougeL": compute_rouge_l(base_predictions, references),
        "tuned_rougeL": compute_rouge_l(tuned_predictions, references),
        "base_dominant_emotion_acc": compute_label_accuracy(base_predictions, [s["target_record"] for s in samples], "dominant_emotion"),
        "tuned_dominant_emotion_acc": compute_label_accuracy(tuned_predictions, [s["target_record"] for s in samples], "dominant_emotion"),
        "base_valence_acc": compute_label_accuracy(base_predictions, [s["target_record"] for s in samples], "emotional_valence"),
        "tuned_valence_acc": compute_label_accuracy(tuned_predictions, [s["target_record"] for s in samples], "emotional_valence"),
        "base_arousal_acc": compute_label_accuracy(base_predictions, [s["target_record"] for s in samples], "emotional_arousal_level"),
        "tuned_arousal_acc": compute_label_accuracy(tuned_predictions, [s["target_record"] for s in samples], "emotional_arousal_level"),
        "base_json_rate": sum(bool(parse_json_response(text)) for text in base_predictions) / max(len(base_predictions), 1),
        "tuned_json_rate": sum(bool(parse_json_response(text)) for text in tuned_predictions) / max(len(tuned_predictions), 1),
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        for sample, base_pred, tuned_pred in zip(samples, base_predictions, tuned_predictions):
            record = {
                "request_id": sample["request_id"],
                "image_path": sample["image_path"],
                "reference": sample["target_record"],
                "base_prediction": base_pred,
                "tuned_prediction": tuned_pred,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = os.path.splitext(args.output_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Results:", args.output_path)
    print("Summary:", summary_path)
    print(summary)


if __name__ == "__main__":
    main()
