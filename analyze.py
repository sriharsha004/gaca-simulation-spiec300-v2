"""
Computes every number the manuscript needs to report, from the REAL raw logs
produced by run_conditions.py / run_ablations.py / run_scalability.py.
No number in this file's output is invented -- if a log file is missing or
empty, the corresponding section is skipped with a printed warning rather
than filled in with a placeholder.

Run after real (non-mock) data collection:
    python3 analyze.py --main results_main.jsonl --ablation results_ablation.jsonl \
        --scalability results_scalability.jsonl --out analysis_report.json
"""

from __future__ import annotations
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"WARNING: {path} not found, skipping.")
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def is_tp(row: dict, judge_threshold: int = 3) -> bool:
    return (row["pred_gap_category_id"] == row["true_gap_category_id"]
            and (row.get("judge_rating") or 0) >= judge_threshold)


def precision_recall_f1(rows: list[dict]) -> tuple[float, float, float]:
    """Per manuscript def: TP = category match AND judge_rating>=3. Computed
    per-category then micro-averaged (every interaction contributes exactly
    one predicted label and one true label -- see docstring in run_conditions.py)."""
    categories = sorted({r["true_gap_category_id"] for r in rows} | {r["pred_gap_category_id"] for r in rows})
    tp = fp = fn = 0
    for r in rows:
        if is_tp(r):
            tp += 1
        else:
            fp += 1
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def rule_compliance_pct(rows: list[dict], strict: bool = True) -> float:
    if not rows:
        return float("nan")
    if strict:
        return 100.0 * sum(1 for r in rows if r["rule_compliance_rate"] >= 1.0) / len(rows)
    return 100.0 * statistics.mean(r["rule_compliance_rate"] for r in rows)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def condition_summary(rows: list[dict]) -> dict:
    precision, recall, f1 = precision_recall_f1(rows)
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    mrrs = [r["mrr_at_10"] for r in rows]
    recalls10 = [r["recall_at_10"] for r in rows]
    ratings = [r["judge_rating"] for r in rows if r.get("judge_rating") is not None]
    return {
        "n": len(rows),
        "precision": precision, "recall": recall, "f1": f1,
        "mrr_at_10": statistics.mean(mrrs) if mrrs else float("nan"),
        "recall_at_10": statistics.mean(recalls10) if recalls10 else float("nan"),
        "judge_rating_mean": statistics.mean(ratings) if ratings else float("nan"),
        "rule_compliance_pct_strict": rule_compliance_pct(rows, strict=True),
        "rule_compliance_pct_partial_credit": rule_compliance_pct(rows, strict=False),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "any_mock": any(r.get("mock") for r in rows),
    }


def main_table(main_rows: list[dict]) -> dict:
    by_condition = defaultdict(list)
    for r in main_rows:
        by_condition[r["condition"]].append(r)
    return {cond: condition_summary(rows) for cond, rows in by_condition.items()}


def domain_table(main_rows: list[dict]) -> dict:
    by_domain_cond = defaultdict(list)
    for r in main_rows:
        by_domain_cond[(r["domain"], r["condition"])].append(r)
    out = defaultdict(dict)
    for (domain, cond), rows in by_domain_cond.items():
        _, _, f1 = precision_recall_f1(rows)
        out[domain][cond] = f1
    return dict(out)


def coefficient_of_variation(values: list[float]) -> float:
    if not values or statistics.mean(values) == 0:
        return float("nan")
    return statistics.pstdev(values) / statistics.mean(values)


def wilcoxon_gaca_vs_baselines(main_rows: list[dict]) -> dict:
    """Paired Wilcoxon signed-rank test on judge_rating, GACA vs each baseline,
    matched by interaction_id. Reports full test statistics + Bonferroni
    correction detail, not just a p-value threshold (editor requirement)."""
    by_key = defaultdict(dict)  # interaction_id -> {condition: judge_rating}
    for r in main_rows:
        if r.get("judge_rating") is not None:
            by_key[r["interaction_id"]][r["condition"]] = r["judge_rating"]

    baselines = ["VL", "FR", "GR"]
    n_comparisons = len(baselines)
    corrected_alpha = 0.05 / n_comparisons
    results = {}
    for baseline in baselines:
        gaca_vals, base_vals = [], []
        for iid, d in by_key.items():
            if "GACA" in d and baseline in d:
                gaca_vals.append(d["GACA"])
                base_vals.append(d[baseline])
        if len(gaca_vals) < 10:
            results[baseline] = {"n_pairs": len(gaca_vals), "note": "insufficient paired samples (<10)"}
            continue
        diffs = np.array(gaca_vals) - np.array(base_vals)
        if np.all(diffs == 0):
            results[baseline] = {"n_pairs": len(gaca_vals), "note": "all differences zero"}
            continue
        try:
            statistic, p_value = stats.wilcoxon(gaca_vals, base_vals, zero_method="wilcox")
        except ValueError as e:
            results[baseline] = {"n_pairs": len(gaca_vals), "error": str(e)}
            continue
        results[baseline] = {
            "n_pairs": len(gaca_vals),
            "wilcoxon_statistic": float(statistic),
            "p_value_raw": float(p_value),
            "significant_after_bonferroni": bool(p_value < corrected_alpha),
        }
    return {
        "n_comparisons": n_comparisons,
        "family_wise_alpha": 0.05,
        "bonferroni_corrected_alpha": corrected_alpha,
        "per_baseline": results,
    }


def ablation_table(ablation_rows: list[dict], gaca_rows_main: list[dict]) -> dict:
    _, _, full_f1 = precision_recall_f1(gaca_rows_main)
    full_compliance = rule_compliance_pct(gaca_rows_main, strict=True)
    out = {"full_gaca": {"f1": full_f1, "rule_compliance_pct": full_compliance}}
    by_config = defaultdict(list)
    for r in ablation_rows:
        by_config[r["config"]].append(r)
    for config, rows in by_config.items():
        _, _, f1 = precision_recall_f1(rows)
        compliance = rule_compliance_pct(rows, strict=True)
        delta_f1_pct = 100.0 * (f1 - full_f1) / full_f1 if full_f1 else float("nan")
        out[config] = {"f1": f1, "rule_compliance_pct": compliance, "delta_f1_pct": delta_f1_pct}
    return out


def scalability_table(scal_rows: list[dict]) -> dict:
    if len(scal_rows) < 2:
        return {"note": "insufficient data points for regression"}
    sizes = np.array([r["actual_size"] for r in scal_rows], dtype=float)
    extraction = np.array([r["extraction_ms"] for r in scal_rows], dtype=float)
    log_sizes = np.log(sizes)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, extraction)
    return {
        "points": scal_rows,
        "log_n_fit": {"slope": slope, "intercept": intercept, "r_squared": r_value ** 2, "p_value": p_value},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="results_main.jsonl")
    ap.add_argument("--ablation", default="results_ablation.jsonl")
    ap.add_argument("--scalability", default="results_scalability.jsonl")
    ap.add_argument("--out", default="analysis_report.json")
    args = ap.parse_args()

    main_rows = load_jsonl(Path(args.main))
    ablation_rows = load_jsonl(Path(args.ablation))
    scal_rows = load_jsonl(Path(args.scalability))

    if any(r.get("mock") for r in main_rows):
        print("*** WARNING: results contain MOCK data (no API key was set during collection). ***")
        print("*** These numbers are placeholders for pipeline verification only and MUST NOT ***")
        print("*** be reported in the manuscript. Re-run with a real OPENAI_API_KEY.          ***")

    report = {
        "table3_main_comparative": main_table(main_rows),
        "table4_domain_f1": domain_table(main_rows),
        "wilcoxon_bonferroni": wilcoxon_gaca_vs_baselines(main_rows),
        "table5_ablation": ablation_table(ablation_rows, [r for r in main_rows if r["condition"] == "GACA"]),
        "table6_scalability": scalability_table(scal_rows),
    }

    domain_f1 = report["table4_domain_f1"]
    gaca_f1_by_domain = [v.get("GACA") for v in domain_f1.values() if v.get("GACA") is not None]
    report["cross_domain_gaca_f1_cv"] = coefficient_of_variation(gaca_f1_by_domain)

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
