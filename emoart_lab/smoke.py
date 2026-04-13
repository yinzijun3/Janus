import argparse
import gc
import json
import time
from pathlib import Path

import torch
from janus.models import VLChatProcessor
from transformers import AutoModelForCausalLM


def run_local_model_smoke_check(model_path: Path) -> dict:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    processor = VLChatProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    summary = {
        "status": "ok",
        "model_path": str(model_path),
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processor_class": processor.__class__.__name__,
        "tokenizer_class": processor.tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "model_type": getattr(model.config, "model_type", None),
        "vocab_size": getattr(processor.tokenizer, "vocab_size", None),
    }
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local-only Janus model smoke check.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_local_model_smoke_check(Path(args.model_path))
    except Exception as exc:  # pragma: no cover - surfaced via CLI/logs
        summary = {
            "status": "failed",
            "model_path": args.model_path,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": repr(exc),
        }
        output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
