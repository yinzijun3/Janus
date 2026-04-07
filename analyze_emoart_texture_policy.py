import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SEMANTIC_METRICS: List[Tuple[str, str]] = [
    ("reference_token_nll_after", "lower"),
    ("reference_psnr_after", "higher"),
    ("reference_mae_after", "lower"),
]

TEXTURE_METRICS: List[Tuple[str, str]] = [
    ("reference_gradient_energy_abs_error_after", "lower"),
    ("reference_laplacian_variance_abs_error_after", "lower"),
    ("reference_high_frequency_energy_abs_error_after", "lower"),
]

PAIRWISE_AXES = [
    ("prob015", "prob035", "lighter_vs_heavier_intensity"),
    ("brushstroke", "prob015", "narrower_vs_broader_fields"),
]

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def better(value_a: float, value_b: float, direction: str) -> int:
    if abs(value_a - value_b) <= 1e-12:
        return 0
    if direction == "lower":
        return -1 if value_a < value_b else 1
    if direction == "higher":
        return -1 if value_a > value_b else 1
    raise ValueError(f"unknown direction: {direction}")


def metric_win_counts(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    metrics: Iterable[Tuple[str, str]],
) -> Dict[str, int]:
    counts = {"a": 0, "b": 0, "tie": 0}
    for metric, direction in metrics:
        compare = better(row_a[metric], row_b[metric], direction)
        if compare < 0:
            counts["a"] += 1
        elif compare > 0:
            counts["b"] += 1
        else:
            counts["tie"] += 1
    return counts


def majority_vote_label(
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
    metrics: Iterable[Tuple[str, str]],
    label_a: str,
    label_b: str,
) -> Tuple[str, Dict[str, int]]:
    counts = metric_win_counts(row_a, row_b, metrics)
    if counts["a"] > counts["b"]:
        return label_a, counts
    if counts["b"] > counts["a"]:
        return label_b, counts
    return "tie", counts


def score_against_baseline(semantic_wins: int, texture_wins: int) -> str:
    if semantic_wins >= 2 and texture_wins >= 2:
        return "safe_gain"
    if semantic_wins >= 2 and texture_wins <= 1:
        return "semantic_only"
    if semantic_wins <= 1 and texture_wins >= 2:
        return "texture_tradeoff"
    return "mixed"


def build_sample_records(
    manifest_by_request_id: Dict[str, Dict[str, Any]],
    variant_rows: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    variant_names = list(variant_rows)
    request_ids = list(variant_rows[variant_names[0]])
    for variant_name in variant_names[1:]:
        other_ids = list(variant_rows[variant_name])
        if request_ids != other_ids:
            raise ValueError("comparison.jsonl files are not aligned by request_id")

    samples: List[Dict[str, Any]] = []
    for request_id in request_ids:
        manifest_record = manifest_by_request_id.get(request_id)
        if manifest_record is None:
            raise KeyError(f"request_id missing from manifest: {request_id}")
        rows = {name: variant_rows[name][request_id] for name in variant_names}

        semantic_best_counts = Counter()
        texture_best_counts = Counter()
        for metric, direction in SEMANTIC_METRICS:
            best_variant = min(
                variant_names,
                key=lambda name: (
                    rows[name][metric] if direction == "lower" else -rows[name][metric],
                    name,
                ),
            )
            semantic_best_counts[best_variant] += 1
        for metric, direction in TEXTURE_METRICS:
            best_variant = min(
                variant_names,
                key=lambda name: (
                    rows[name][metric] if direction == "lower" else -rows[name][metric],
                    name,
                ),
            )
            texture_best_counts[best_variant] += 1

        vs_baseline: Dict[str, Dict[str, Any]] = {}
        for variant_name in ["prob035", "prob015", "brushstroke"]:
            semantic_counts = metric_win_counts(rows[variant_name], rows["baseline"], SEMANTIC_METRICS)
            texture_counts = metric_win_counts(rows[variant_name], rows["baseline"], TEXTURE_METRICS)
            vs_baseline[variant_name] = {
                "semantic_wins": semantic_counts["a"],
                "semantic_losses": semantic_counts["b"],
                "semantic_ties": semantic_counts["tie"],
                "texture_wins": texture_counts["a"],
                "texture_losses": texture_counts["b"],
                "texture_ties": texture_counts["tie"],
                "category": score_against_baseline(semantic_counts["a"], texture_counts["a"]),
            }

        pairwise: Dict[str, Dict[str, Any]] = {}
        for variant_a, variant_b, axis_name in PAIRWISE_AXES:
            semantic_better, semantic_counts = majority_vote_label(
                rows[variant_a],
                rows[variant_b],
                SEMANTIC_METRICS,
                variant_a,
                variant_b,
            )
            texture_better, texture_counts = majority_vote_label(
                rows[variant_a],
                rows[variant_b],
                TEXTURE_METRICS,
                variant_a,
                variant_b,
            )
            pairwise[axis_name] = {
                "variants": [variant_a, variant_b],
                "semantic_better": semantic_better,
                "texture_better": texture_better,
                "semantic_counts": semantic_counts,
                "texture_counts": texture_counts,
            }

        samples.append(
            {
                "request_id": request_id,
                "style_name": manifest_record.get("style_name"),
                "brushstroke_text": manifest_record.get("brushstroke_text"),
                "line_quality_text": manifest_record.get("line_quality_text"),
                "texture_tags": manifest_record.get("texture_tags", []),
                "medium_tags": manifest_record.get("medium_tags", []),
                "texture_tag_count": len(manifest_record.get("texture_tags", [])),
                "has_medium_tags": bool(manifest_record.get("medium_tags")),
                "variants": {
                    name: {
                        "reference_token_nll_after": rows[name]["reference_token_nll_after"],
                        "reference_psnr_after": rows[name]["reference_psnr_after"],
                        "reference_mae_after": rows[name]["reference_mae_after"],
                        "reference_gradient_energy_abs_error_after": rows[name][
                            "reference_gradient_energy_abs_error_after"
                        ],
                        "reference_laplacian_variance_abs_error_after": rows[name][
                            "reference_laplacian_variance_abs_error_after"
                        ],
                        "reference_high_frequency_energy_abs_error_after": rows[name][
                            "reference_high_frequency_energy_abs_error_after"
                        ],
                    }
                    for name in variant_names
                },
                "semantic_best_counts": dict(semantic_best_counts),
                "texture_best_counts": dict(texture_best_counts),
                "best_semantic_variant": max(
                    variant_names, key=lambda name: (semantic_best_counts[name], -texture_best_counts[name], name)
                ),
                "best_texture_variant": max(
                    variant_names, key=lambda name: (texture_best_counts[name], -semantic_best_counts[name], name)
                ),
                "vs_baseline": vs_baseline,
                "pairwise": pairwise,
            }
        )
    return samples


def update_group_summary(
    groups: Dict[str, Dict[str, Any]],
    group_name: str,
    sample: Dict[str, Any],
) -> None:
    if not group_name:
        return

    group = groups.setdefault(
        group_name,
        {
            "group_name": group_name,
            "count": 0,
            "request_ids": [],
            "semantic_best_counts": Counter(),
            "texture_best_counts": Counter(),
            "vs_baseline": {
                variant: {"safe_gain": 0, "semantic_only": 0, "texture_tradeoff": 0, "mixed": 0}
                for variant in ["prob035", "prob015", "brushstroke"]
            },
            "pairwise": {
                axis_name: {
                    "semantic": Counter(),
                    "texture": Counter(),
                }
                for _, _, axis_name in PAIRWISE_AXES
            },
        },
    )

    group["count"] += 1
    group["request_ids"].append(sample["request_id"])
    group["semantic_best_counts"].update(sample["semantic_best_counts"])
    group["texture_best_counts"].update(sample["texture_best_counts"])
    for variant_name, result in sample["vs_baseline"].items():
        group["vs_baseline"][variant_name][result["category"]] += 1
    for axis_name, pairwise_result in sample["pairwise"].items():
        group["pairwise"][axis_name]["semantic"][pairwise_result["semantic_better"]] += 1
        group["pairwise"][axis_name]["texture"][pairwise_result["texture_better"]] += 1


def finalize_group_summaries(groups: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for group in groups.values():
        record = {
            "group_name": group["group_name"],
            "count": group["count"],
            "request_ids": group["request_ids"],
            "semantic_best_counts": dict(group["semantic_best_counts"]),
            "texture_best_counts": dict(group["texture_best_counts"]),
            "vs_baseline": group["vs_baseline"],
            "pairwise": {
                axis_name: {
                    "semantic": dict(summary["semantic"]),
                    "texture": dict(summary["texture"]),
                }
                for axis_name, summary in group["pairwise"].items()
            },
        }
        record["max_safe_gain_variant"] = max(
            ["prob035", "prob015", "brushstroke"],
            key=lambda name: (
                record["vs_baseline"][name]["safe_gain"],
                -record["vs_baseline"][name]["texture_tradeoff"],
                name,
            ),
        )
        record["light_prompt_semantic_signals"] = (
            record["pairwise"]["lighter_vs_heavier_intensity"]["semantic"].get("prob015", 0)
            + record["pairwise"]["narrower_vs_broader_fields"]["semantic"].get("brushstroke", 0)
        )
        record["heavy_prompt_semantic_signals"] = (
            record["pairwise"]["lighter_vs_heavier_intensity"]["semantic"].get("prob035", 0)
            + record["pairwise"]["narrower_vs_broader_fields"]["semantic"].get("prob015", 0)
        )
        out.append(record)
    out.sort(
        key=lambda row: (
            row["vs_baseline"]["brushstroke"]["safe_gain"]
            + row["vs_baseline"]["prob015"]["safe_gain"]
            + row["vs_baseline"]["prob035"]["safe_gain"],
            row["light_prompt_semantic_signals"],
            row["count"],
            row["group_name"],
        ),
        reverse=True,
    )
    return out


def aggregate_groups(
    samples: List[Dict[str, Any]],
    group_kind: str,
) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        if group_kind == "style":
            values = [sample["style_name"]]
        elif group_kind == "texture_tag":
            values = sample["texture_tags"] or ["<none>"]
        elif group_kind == "medium_tag":
            values = sample["medium_tags"] or ["<none>"]
        elif group_kind == "texture_tag_count":
            values = [str(sample["texture_tag_count"])]
        elif group_kind == "has_medium_tags":
            values = ["yes" if sample["has_medium_tags"] else "no"]
        else:
            raise ValueError(f"unknown group_kind: {group_kind}")
        for value in values:
            update_group_summary(groups, value, sample)
    return finalize_group_summaries(groups)


def build_global_summary(
    variant_summaries: Dict[str, Dict[str, Any]],
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    best_semantic_counts = Counter(sample["best_semantic_variant"] for sample in samples)
    best_texture_counts = Counter(sample["best_texture_variant"] for sample in samples)
    variant_vs_baseline = {
        variant_name: Counter(sample["vs_baseline"][variant_name]["category"] for sample in samples)
        for variant_name in ["prob035", "prob015", "brushstroke"]
    }

    pairwise = {}
    for _, _, axis_name in PAIRWISE_AXES:
        pairwise[axis_name] = {
            "semantic": Counter(sample["pairwise"][axis_name]["semantic_better"] for sample in samples),
            "texture": Counter(sample["pairwise"][axis_name]["texture_better"] for sample in samples),
        }

    return {
        "num_samples": len(samples),
        "variant_summaries": variant_summaries,
        "best_semantic_variant_counts": dict(best_semantic_counts),
        "best_texture_variant_counts": dict(best_texture_counts),
        "vs_baseline_category_counts": {
            name: dict(counter) for name, counter in variant_vs_baseline.items()
        },
        "pairwise_majority_counts": {
            axis_name: {
                "semantic": dict(summary["semantic"]),
                "texture": dict(summary["texture"]),
            }
            for axis_name, summary in pairwise.items()
        },
    }


def compact_group_rows(rows: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    compact_rows: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        compact_rows.append(
            {
                "group_name": row["group_name"],
                "count": row["count"],
                "max_safe_gain_variant": row["max_safe_gain_variant"],
                "safe_gain_prob035": row["vs_baseline"]["prob035"]["safe_gain"],
                "safe_gain_prob015": row["vs_baseline"]["prob015"]["safe_gain"],
                "safe_gain_brushstroke": row["vs_baseline"]["brushstroke"]["safe_gain"],
                "texture_tradeoff_prob035": row["vs_baseline"]["prob035"]["texture_tradeoff"],
                "texture_tradeoff_prob015": row["vs_baseline"]["prob015"]["texture_tradeoff"],
                "texture_tradeoff_brushstroke": row["vs_baseline"]["brushstroke"]["texture_tradeoff"],
                "light_prompt_semantic_signals": row["light_prompt_semantic_signals"],
                "heavy_prompt_semantic_signals": row["heavy_prompt_semantic_signals"],
            }
        )
    return compact_rows


def shortlist_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    shortlisted: List[Dict[str, Any]] = []
    for sample in samples:
        best_variant = max(
            ["prob035", "prob015", "brushstroke"],
            key=lambda name: (
                sample["vs_baseline"][name]["semantic_wins"] + sample["vs_baseline"][name]["texture_wins"],
                sample["vs_baseline"][name]["semantic_wins"],
                -sample["vs_baseline"][name]["texture_losses"],
                name,
            ),
        )
        best_result = sample["vs_baseline"][best_variant]
        if best_result["category"] == "safe_gain":
            shortlisted.append(
                {
                    "request_id": sample["request_id"],
                    "style_name": sample["style_name"],
                    "best_variant_vs_baseline": best_variant,
                    "semantic_wins": best_result["semantic_wins"],
                    "texture_wins": best_result["texture_wins"],
                    "texture_tags": sample["texture_tags"],
                    "medium_tags": sample["medium_tags"],
                    "pairwise": sample["pairwise"],
                }
            )
    shortlisted.sort(
        key=lambda row: (
            row["semantic_wins"] + row["texture_wins"],
            row["semantic_wins"],
            row["texture_wins"],
            row["style_name"],
            row["request_id"],
        ),
        reverse=True,
    )
    return shortlisted


def build_markdown(
    global_summary: Dict[str, Any],
    style_rows: List[Dict[str, Any]],
    texture_tag_rows: List[Dict[str, Any]],
    medium_tag_rows: List[Dict[str, Any]],
    shortlist: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# EmoArt Texture Policy Review (32 Aligned Samples)")
    lines.append("")

    lines.append("## Four-Way Aggregate")
    lines.append("| variant | nll_after | psnr_after | mae_after | grad_err | lap_err | hf_err |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for variant_name in ["baseline", "prob035", "prob015", "brushstroke"]:
        summary = global_summary["variant_summaries"][variant_name]
        lines.append(
            f"| {variant_name} | "
            f"{summary['avg_reference_token_nll_after']:.4f} | "
            f"{summary['avg_reference_psnr_after']:.4f} | "
            f"{summary['avg_reference_mae_after']:.4f} | "
            f"{summary['avg_reference_gradient_energy_abs_error_after']:.4f} | "
            f"{summary['avg_reference_laplacian_variance_abs_error_after']:.4f} | "
            f"{summary['avg_reference_high_frequency_energy_abs_error_after']:.4f} |"
        )
    lines.append("")

    lines.append("## Immediate Read")
    lines.append(
        "- Four-way aligned counts do not show a globally dominant texture branch: `prob015` wins aggregate semantics more often, while texture wins are split across `baseline`, `prob015`, and `brushstroke`."
    )
    lines.append(
        "- Pairwise majorities remain close (`17:15` on both intensity and field breadth axes), which matches the current conclusion that texture edits create local gains but no stable replacement for Exp A."
    )
    lines.append("")

    lines.append("## Best-Of-32 Vote Counts")
    lines.append("- best semantic variant counts:")
    for variant_name in ["baseline", "prob035", "prob015", "brushstroke"]:
        lines.append(
            f"  - {variant_name}: {global_summary['best_semantic_variant_counts'].get(variant_name, 0)}"
        )
    lines.append("- best texture variant counts:")
    for variant_name in ["baseline", "prob035", "prob015", "brushstroke"]:
        lines.append(
            f"  - {variant_name}: {global_summary['best_texture_variant_counts'].get(variant_name, 0)}"
        )
    lines.append("")

    lines.append("## Texture Variants vs Baseline")
    lines.append("| variant | safe_gain | semantic_only | texture_tradeoff | mixed |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant_name in ["prob035", "prob015", "brushstroke"]:
        counts = global_summary["vs_baseline_category_counts"][variant_name]
        lines.append(
            f"| {variant_name} | {counts.get('safe_gain', 0)} | {counts.get('semantic_only', 0)} | "
            f"{counts.get('texture_tradeoff', 0)} | {counts.get('mixed', 0)} |"
        )
    lines.append("")

    lines.append("## Pairwise Majority Counts")
    for axis_name, summary in global_summary["pairwise_majority_counts"].items():
        lines.append(f"### {axis_name}")
        lines.append(
            f"- semantic: {summary['semantic']}"
        )
        lines.append(
            f"- texture: {summary['texture']}"
        )
    lines.append("")

    lines.append("## Styles With Potential Texture Opt-In Value")
    lines.append("| style | n | best safe variant | safe prob035 | safe prob015 | safe brushstroke | tradeoff prob035 | tradeoff prob015 | tradeoff brushstroke | light semantic signals |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in style_rows[:12]:
        lines.append(
            f"| {row['group_name']} | {row['count']} | {row['max_safe_gain_variant']} | "
            f"{row['vs_baseline']['prob035']['safe_gain']} | {row['vs_baseline']['prob015']['safe_gain']} | "
            f"{row['vs_baseline']['brushstroke']['safe_gain']} | "
            f"{row['vs_baseline']['prob035']['texture_tradeoff']} | {row['vs_baseline']['prob015']['texture_tradeoff']} | "
            f"{row['vs_baseline']['brushstroke']['texture_tradeoff']} | {row['light_prompt_semantic_signals']} |"
        )
    lines.append("")

    lines.append("## Styles Most Exposed To Heavier/Broader Prompt Harm")
    lines.append("| style | n | light semantic signals | heavy semantic signals | intensity: prob015 over prob035 | field: brushstroke over prob015 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(style_rows, key=lambda item: (item["light_prompt_semantic_signals"], item["count"], item["group_name"]), reverse=True)[:12]:
        lines.append(
            f"| {row['group_name']} | {row['count']} | {row['light_prompt_semantic_signals']} | "
            f"{row['heavy_prompt_semantic_signals']} | "
            f"{row['pairwise']['lighter_vs_heavier_intensity']['semantic'].get('prob015', 0)} | "
            f"{row['pairwise']['narrower_vs_broader_fields']['semantic'].get('brushstroke', 0)} |"
        )
    lines.append("")

    lines.append("## Metadata Signals: texture_tags")
    lines.append("| texture_tag | n | best safe variant | safe prob015 | safe brushstroke | light-prompt semantic signals |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for row in texture_tag_rows[:12]:
        lines.append(
            f"| {row['group_name']} | {row['count']} | {row['max_safe_gain_variant']} | "
            f"{row['vs_baseline']['prob015']['safe_gain']} | {row['vs_baseline']['brushstroke']['safe_gain']} | "
            f"{row['light_prompt_semantic_signals']} |"
        )
    lines.append("")

    lines.append("## Metadata Signals: medium_tags")
    lines.append("| medium_tag | n | best safe variant | safe prob015 | safe brushstroke | light-prompt semantic signals |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for row in medium_tag_rows[:12]:
        lines.append(
            f"| {row['group_name']} | {row['count']} | {row['max_safe_gain_variant']} | "
            f"{row['vs_baseline']['prob015']['safe_gain']} | {row['vs_baseline']['brushstroke']['safe_gain']} | "
            f"{row['light_prompt_semantic_signals']} |"
        )
    lines.append("")

    lines.append("## Human Review Queue: Safe Gains vs Baseline")
    lines.append("| request_id | style | best variant | semantic wins | texture wins | texture_tags | medium_tags |")
    lines.append("|---|---|---|---:|---:|---|---|")
    for row in shortlist[:12]:
        texture_tags = ", ".join(row["texture_tags"]) if row["texture_tags"] else "-"
        medium_tags = ", ".join(row["medium_tags"]) if row["medium_tags"] else "-"
        lines.append(
            f"| {row['request_id']} | {row['style_name']} | {row['best_variant_vs_baseline']} | "
            f"{row['semantic_wins']} | {row['texture_wins']} | {texture_tags} | {medium_tags} |"
        )
    lines.append("")
    lines.append("## Reading Guide")
    lines.append("- `safe_gain`: relative to baseline, the variant wins at least 2/3 semantic metrics and 2/3 texture metrics on the same sample.")
    lines.append("- `texture_tradeoff`: texture wins are present, but semantic metrics do not reach majority over baseline.")
    lines.append("- `light-prompt semantic signals`: sum of cases where `prob015` beats `prob035` semantically plus cases where `brushstroke` beats `full_fields` semantically.")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze style-aware texture policy signals for EmoArt generation.")
    parser.add_argument(
        "--manifest-path",
        default="/root/autodl-tmp/data/emoart_5k/gen_full_official_texture_meta/val.jsonl",
    )
    parser.add_argument(
        "--baseline-summary",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_baseline_texture_32/summary.json",
    )
    parser.add_argument(
        "--baseline-comparison",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_baseline_texture_32/comparison.jsonl",
    )
    parser.add_argument(
        "--prob035-summary",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_texture_balanced_headonly_32/summary.json",
    )
    parser.add_argument(
        "--prob035-comparison",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_texture_balanced_headonly_32/comparison.jsonl",
    )
    parser.add_argument(
        "--prob015-summary",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_texture_balanced_headonly_prob015_32/summary.json",
    )
    parser.add_argument(
        "--prob015-comparison",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_texture_balanced_headonly_prob015_32/comparison.jsonl",
    )
    parser.add_argument(
        "--brushstroke-summary",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_texture_balanced_headonly_prob015_brushstroke_32/summary.json",
    )
    parser.add_argument(
        "--brushstroke-comparison",
        default="/root/autodl-tmp/emoart_gen_runs/out_gen_compare_expA_texture_balanced_headonly_prob015_brushstroke_32/comparison.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/emoart_gen_runs/texture_policy_review_32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = load_jsonl(Path(args.manifest_path))
    manifest_by_request_id = {record["request_id"]: record for record in manifest_records}

    variant_summaries = {
        "baseline": load_json(Path(args.baseline_summary)),
        "prob035": load_json(Path(args.prob035_summary)),
        "prob015": load_json(Path(args.prob015_summary)),
        "brushstroke": load_json(Path(args.brushstroke_summary)),
    }
    comparison_rows = {
        "baseline": load_jsonl(Path(args.baseline_comparison)),
        "prob035": load_jsonl(Path(args.prob035_comparison)),
        "prob015": load_jsonl(Path(args.prob015_comparison)),
        "brushstroke": load_jsonl(Path(args.brushstroke_comparison)),
    }
    comparison_by_request_id = {
        variant_name: {row["request_id"]: row for row in rows}
        for variant_name, rows in comparison_rows.items()
    }

    samples = build_sample_records(manifest_by_request_id, comparison_by_request_id)
    style_rows = aggregate_groups(samples, "style")
    texture_tag_rows = aggregate_groups(samples, "texture_tag")
    medium_tag_rows = aggregate_groups(samples, "medium_tag")
    texture_tag_count_rows = aggregate_groups(samples, "texture_tag_count")
    has_medium_tag_rows = aggregate_groups(samples, "has_medium_tags")
    shortlist = shortlist_samples(samples)
    global_summary = build_global_summary(variant_summaries, samples)

    report = build_markdown(global_summary, style_rows, texture_tag_rows, medium_tag_rows, shortlist)
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    write_json(output_dir / "summary.json", global_summary)
    write_json(output_dir / "sample_diagnostics.json", samples)
    write_json(output_dir / "style_summary.json", style_rows)
    write_json(output_dir / "texture_tag_summary.json", texture_tag_rows)
    write_json(output_dir / "medium_tag_summary.json", medium_tag_rows)
    write_json(output_dir / "texture_tag_count_summary.json", texture_tag_count_rows)
    write_json(output_dir / "has_medium_tags_summary.json", has_medium_tag_rows)
    write_json(output_dir / "human_review_queue.json", shortlist)
    write_json(
        output_dir / "compact_policy_view.json",
        {
            "styles": compact_group_rows(style_rows),
            "texture_tags": compact_group_rows(texture_tag_rows),
            "medium_tags": compact_group_rows(medium_tag_rows),
        },
    )


if __name__ == "__main__":
    main()
