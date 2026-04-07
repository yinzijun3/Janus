import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def style_from_request_id(request_id: str) -> str:
    return re.sub(r"_request-\d+$", "", request_id)


def train_run_summary(path: Path) -> Dict[str, Any]:
    config = load_json(path / "train_config.json")
    rows = load_jsonl(path / "train_log.jsonl")
    eval_rows = [row for row in rows if "eval_loss" in row]
    tail_rows = rows[-5:]
    final_metrics = {
        "final_global_step": rows[-1]["global_step"] if rows else None,
        "last_logged_loss": rows[-1].get("loss") if rows else None,
        "last_logged_token_accuracy": rows[-1].get("token_accuracy") if rows else None,
    }
    if eval_rows:
        final_metrics["eval_history"] = eval_rows
        final_metrics["best_eval_loss"] = min(row["eval_loss"] for row in eval_rows)
        token_acc_values = [
            row["eval_token_accuracy"]
            for row in eval_rows
            if "eval_token_accuracy" in row
        ]
        if token_acc_values:
            final_metrics["best_eval_token_accuracy"] = max(token_acc_values)
    return {
        "name": path.name,
        "path": str(path),
        "config": config,
        "final_metrics": final_metrics,
        "tail_rows": tail_rows,
        "checkpoints": sorted(p.name for p in path.glob("checkpoint-*")),
        "has_final_adapter": (path / "final_adapter").exists(),
    }


def compare_run_summary(path: Path) -> Dict[str, Any]:
    summary = load_json(path / "summary.json")
    rows = load_jsonl(path / "comparison.jsonl")
    return {
        "name": path.name,
        "path": str(path),
        "summary": summary,
        "num_rows": len(rows),
        "request_ids": [row["request_id"] for row in rows],
    }


def style_level_summary(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[style_from_request_id(row["request_id"])].append(row)

    out: List[Dict[str, Any]] = []
    for style, items in groups.items():
        record: Dict[str, Any] = {
            "style": style,
            "num_samples": len(items),
            "avg_nll_gain": mean(
                item["reference_token_nll_before"] - item["reference_token_nll_after"]
                for item in items
            ),
            "avg_psnr_gain": mean(
                item["reference_psnr_after"] - item["reference_psnr_before"] for item in items
            ),
            "avg_mae_gain": mean(
                item["reference_mae_before"] - item["reference_mae_after"] for item in items
            ),
        }
        if "reference_gradient_energy_abs_error_after" in items[0]:
            record.update(
                {
                    "avg_gradient_gain": mean(
                        item["reference_gradient_energy_abs_error_before"]
                        - item["reference_gradient_energy_abs_error_after"]
                        for item in items
                    ),
                    "avg_laplacian_gain": mean(
                        item["reference_laplacian_variance_abs_error_before"]
                        - item["reference_laplacian_variance_abs_error_after"]
                        for item in items
                    ),
                    "avg_high_frequency_gain": mean(
                        item["reference_high_frequency_energy_abs_error_before"]
                        - item["reference_high_frequency_energy_abs_error_after"]
                        for item in items
                    ),
                    "texture_vote_count": sum(
                        int(item["improved_gradient_energy"])
                        + int(item["improved_laplacian_variance"])
                        + int(item["improved_high_frequency_energy"])
                        for item in items
                    ),
                }
            )
            record["texture_composite"] = (
                record["avg_gradient_gain"]
                + record["avg_laplacian_gain"] / 100.0
                + record["avg_high_frequency_gain"] / 100.0
            )
        out.append(record)
    out.sort(key=lambda row: (row.get("texture_composite", 0.0), row["avg_nll_gain"]), reverse=True)
    return out


def pairwise_compare(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    if [row["request_id"] for row in rows_a] != [row["request_id"] for row in rows_b]:
        raise ValueError("pairwise_compare requires aligned request_id ordering")

    metrics: List[Tuple[str, str]] = [
        ("reference_token_nll_after", "lower"),
        ("reference_psnr_after", "higher"),
        ("reference_mae_after", "lower"),
    ]
    if "reference_gradient_energy_abs_error_after" in rows_a[0]:
        metrics.extend(
            [
                ("reference_gradient_energy_abs_error_after", "lower"),
                ("reference_laplacian_variance_abs_error_after", "lower"),
                ("reference_high_frequency_energy_abs_error_after", "lower"),
            ]
        )

    metric_wins: Dict[str, Dict[str, int]] = {}
    samples: List[Dict[str, Any]] = []
    for metric, direction in metrics:
        metric_wins[metric] = {"run_a": 0, "run_b": 0, "tie": 0}

    for row_a, row_b in zip(rows_a, rows_b):
        sample: Dict[str, Any] = {
            "request_id": row_a["request_id"],
            "style": style_from_request_id(row_a["request_id"]),
        }
        texture_delta = 0.0
        for metric, direction in metrics:
            value_a = row_a[metric]
            value_b = row_b[metric]
            if abs(value_a - value_b) < 1e-12:
                winner = "tie"
                metric_wins[metric]["tie"] += 1
            elif (direction == "lower" and value_b < value_a) or (
                direction == "higher" and value_b > value_a
            ):
                winner = "run_b"
                metric_wins[metric]["run_b"] += 1
            else:
                winner = "run_a"
                metric_wins[metric]["run_a"] += 1
            sample[f"{metric}_winner"] = winner
            sample[f"{metric}_delta_b_minus_a"] = value_b - value_a
            if metric.endswith("abs_error_after"):
                texture_delta += value_a - value_b
        sample["texture_composite_gain_b_vs_a"] = texture_delta
        samples.append(sample)

    samples.sort(key=lambda row: row["texture_composite_gain_b_vs_a"], reverse=True)
    return {
        "metric_wins": metric_wins,
        "top_run_b_texture_wins": samples[:10],
        "top_run_b_texture_losses": samples[-10:],
    }


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown(
    train_runs: Dict[str, Dict[str, Any]],
    compare_runs: Dict[str, Dict[str, Any]],
    headonly_style: List[Dict[str, Any]],
    pairwise: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# EmoArt Run Review")
    lines.append("")
    lines.append("## Train Runs")
    for key in [
        "out_gen_expA_64",
        "out_gen_expA_full",
        "out_gen_expA_texture_balanced_v1",
        "out_gen_expA_texture_balanced_headonly_v1",
    ]:
        run = train_runs.get(key)
        if not run:
            continue
        config = run["config"]
        metrics = run["final_metrics"]
        lines.append(f"### {key}")
        lines.append(f"- output: `{run['path']}`")
        lines.append(
            f"- lora_r={config.get('lora_r')} alpha={config.get('lora_alpha')} "
            f"lr={config.get('learning_rate')} gen_lr={config.get('generation_learning_rate')}"
        )
        lines.append(
            f"- generation_module_mode={config.get('generation_module_mode', 'full')} "
            f"enabled={config.get('enabled_generation_modules')}"
        )
        if config.get("prompt_template") is not None:
            lines.append(
                f"- prompt_template={config.get('prompt_template')} "
                f"art_texture_mode={config.get('art_texture_mode')} "
                f"art_texture_prob={config.get('art_texture_prob')}"
            )
        if "eval_history" in metrics:
            last_eval = metrics["eval_history"][-1]
            lines.append(
                f"- last eval: step={last_eval['global_step']} eval_loss={last_eval['eval_loss']:.4f} "
                f"eval_token_accuracy={last_eval['eval_token_accuracy']:.4f}"
            )
        lines.append(f"- checkpoints: {', '.join(run['checkpoints'])}")
        lines.append("")

    lines.append("## Compare Runs")
    for key in [
        "out_gen_compare_expA_8",
        "out_gen_compare_expA_32",
        "out_gen_compare_expA_64",
        "out_gen_compare_expA_baseline_texture_32",
        "out_gen_compare_expA_texture_balanced_32",
        "out_gen_compare_expA_texture_balanced_headonly_32",
        "out_gen_compare_expA_texture_balanced_headonly_64",
    ]:
        run = compare_runs.get(key)
        if not run:
            continue
        summary = run["summary"]
        lines.append(f"### {key}")
        lines.append(f"- adapter: `{summary.get('adapter_path')}`")
        lines.append(
            f"- samples={summary.get('num_samples')} changed={summary.get('changed_outputs')} "
            f"improved_nll={summary.get('improved_reference_nll_count')}"
        )
        lines.append(
            f"- nll after={summary.get('avg_reference_token_nll_after')} "
            f"psnr after={summary.get('avg_reference_psnr_after')} "
            f"mae after={summary.get('avg_reference_mae_after')}"
        )
        if "avg_reference_gradient_energy_abs_error_after" in summary:
            lines.append(
                "- texture abs errors after="
                f"grad {summary.get('avg_reference_gradient_energy_abs_error_after')}, "
                f"lap {summary.get('avg_reference_laplacian_variance_abs_error_after')}, "
                f"hf {summary.get('avg_reference_high_frequency_energy_abs_error_after')}"
            )
        lines.append("")

    lines.append("## Head-Only 64 Style Signals")
    for row in headonly_style[:10]:
        lines.append(
            f"- best: {row['style']} n={row['num_samples']} "
            f"texture_composite={row.get('texture_composite', 0.0):.2f} "
            f"nll_gain={row['avg_nll_gain']:.3f}"
        )
    for row in headonly_style[-10:]:
        lines.append(
            f"- worst: {row['style']} n={row['num_samples']} "
            f"texture_composite={row.get('texture_composite', 0.0):.2f} "
            f"nll_gain={row['avg_nll_gain']:.3f}"
        )
    lines.append("")

    lines.append("## Baseline vs Head-Only on Same 32 Samples")
    for metric, wins in pairwise["metric_wins"].items():
        lines.append(f"- {metric}: baseline={wins['run_a']} head_only={wins['run_b']} tie={wins['tie']}")
    lines.append("")
    lines.append("### Strong Head-Only Texture Wins")
    for row in pairwise["top_run_b_texture_wins"][:5]:
        lines.append(
            f"- {row['request_id']} ({row['style']}): "
            f"texture_gain={row['texture_composite_gain_b_vs_a']:.2f}"
        )
    lines.append("")
    lines.append("### Strong Head-Only Texture Losses")
    for row in pairwise["top_run_b_texture_losses"][-5:]:
        lines.append(
            f"- {row['request_id']} ({row['style']}): "
            f"texture_gain={row['texture_composite_gain_b_vs_a']:.2f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Janus / EmoArt experiment outputs.")
    parser.add_argument(
        "--runs-root",
        default="/root/autodl-tmp/emoart_gen_runs",
        help="Root directory containing out_gen_* experiment folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/emoart_gen_runs/analysis_review",
        help="Directory to write aggregated analysis artifacts.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_runs: Dict[str, Dict[str, Any]] = {}
    compare_runs: Dict[str, Dict[str, Any]] = {}
    for path in sorted(runs_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "train_config.json").exists() and (path / "train_log.jsonl").exists():
            train_runs[path.name] = train_run_summary(path)
        if (path / "summary.json").exists() and (path / "comparison.jsonl").exists():
            compare_runs[path.name] = compare_run_summary(path)

    headonly_rows = load_jsonl(
        runs_root / "out_gen_compare_expA_texture_balanced_headonly_64" / "comparison.jsonl"
    )
    baseline_rows = load_jsonl(
        runs_root / "out_gen_compare_expA_baseline_texture_32" / "comparison.jsonl"
    )
    headonly_32_rows = load_jsonl(
        runs_root / "out_gen_compare_expA_texture_balanced_headonly_32" / "comparison.jsonl"
    )

    headonly_style = style_level_summary(headonly_rows)
    pairwise = pairwise_compare(baseline_rows, headonly_32_rows)

    write_json(output_dir / "train_runs.json", train_runs)
    write_json(output_dir / "compare_runs.json", compare_runs)
    write_json(output_dir / "headonly_64_style_summary.json", headonly_style)
    write_json(output_dir / "baseline_texture32_vs_headonly32.json", pairwise)
    (output_dir / "review.md").write_text(
        build_markdown(train_runs, compare_runs, headonly_style, pairwise),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
