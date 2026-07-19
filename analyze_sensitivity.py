"""
Summarizes degraded-input sensitivity results from results_sensitivity.jsonl.

This script does not call an LLM. It recomputes reportable metrics from the
raw sensitivity log and compares each degraded GACA-only configuration against
the full GACA rows already present in results_main.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze import load_jsonl, percentile, precision_recall_f1, rule_compliance_pct


def condition_summary(rows: list[dict]) -> dict:
    precision, recall, f1 = precision_recall_f1(rows)
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    wall_clock = [r["wall_clock_ms"] for r in rows if r.get("wall_clock_ms") is not None]
    mrrs = [r["mrr_at_10"] for r in rows]
    recalls10 = [r["recall_at_10"] for r in rows]
    ratings = [r["judge_rating"] for r in rows if r.get("judge_rating") is not None]
    mismatches = sum(1 for r in rows if r["pred_gap_category_id"] != r["true_gap_category_id"])
    return {
        "n": len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr_at_10": statistics.mean(mrrs) if mrrs else float("nan"),
        "recall_at_10": statistics.mean(recalls10) if recalls10 else float("nan"),
        "judge_rating_mean": statistics.mean(ratings) if ratings else float("nan"),
        "rule_compliance_pct_strict": rule_compliance_pct(rows, strict=True),
        "rule_compliance_pct_partial_credit": rule_compliance_pct(rows, strict=False),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "wall_clock_p50_ms": percentile(wall_clock, 50),
        "wall_clock_p95_ms": percentile(wall_clock, 95),
        "misclassification_rate_pct": 100.0 * mismatches / len(rows) if rows else float("nan"),
        "any_mock": any(r.get("mock") for r in rows),
    }


def by_config(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    return {config: condition_summary(config_rows) for config, config_rows in sorted(grouped.items())}


def domain_f1_by_config(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["config"], row["domain"])].append(row)
    out = defaultdict(dict)
    for (config, domain), group_rows in grouped.items():
        _, _, f1 = precision_recall_f1(group_rows)
        out[config][domain] = f1
    return {config: dict(domains) for config, domains in sorted(out.items())}


def baseline_gaca_summary(main_rows: list[dict]) -> dict:
    gaca_rows = [r for r in main_rows if r.get("condition") == "GACA"]
    return condition_summary(gaca_rows)


def delta_vs_baseline(config_summaries: dict, baseline: dict) -> dict:
    out = {}
    baseline_f1 = baseline.get("f1")
    baseline_rule = baseline.get("rule_compliance_pct_strict")
    baseline_rating = baseline.get("judge_rating_mean")
    for config, summary in config_summaries.items():
        out[config] = {
            "f1_delta_abs": summary["f1"] - baseline_f1,
            "f1_delta_percentage_points": 100.0 * (summary["f1"] - baseline_f1),
            "rule_compliance_strict_delta_percentage_points": (
                summary["rule_compliance_pct_strict"] - baseline_rule
            ),
            "judge_rating_mean_delta": summary["judge_rating_mean"] - baseline_rating,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensitivity", default="results_sensitivity.jsonl")
    ap.add_argument("--main", default="results_main.jsonl")
    ap.add_argument("--out", default="sensitivity_report.json")
    args = ap.parse_args()

    sensitivity_rows = load_jsonl(Path(args.sensitivity))
    main_rows = load_jsonl(Path(args.main))
    config_summaries = by_config(sensitivity_rows)
    baseline = baseline_gaca_summary(main_rows)
    report = {
        "baseline_gaca_from_results_main": baseline,
        "degraded_input_by_config": config_summaries,
        "delta_vs_baseline_gaca": delta_vs_baseline(config_summaries, baseline),
        "domain_f1_by_config": domain_f1_by_config(sensitivity_rows),
        "row_count": len(sensitivity_rows),
        "any_mock": any(r.get("mock") for r in sensitivity_rows),
    }

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
