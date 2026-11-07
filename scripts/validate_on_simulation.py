"""Check the metrics against known answers before running them on a real judge.

1. Plant a length bias and a position bias in a fake judge, and check we recover them.
2. Check that the 90% interval for kappa really covers the truth about 90% of the time,
   for the cluster bootstrap (by question) and for the naive pair bootstrap.

Usage: python scripts/validate_on_simulation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from judgegate.metrics import kappa_ci, kappa_stat, point_metrics  # noqa: E402
from judgegate.simulate import simulate  # noqa: E402


def planted_bias_table() -> pd.DataFrame:
    rows = []
    for name, kw in [("no bias", {}), ("length bias", {"length_bias": 1.5}),
                     ("position bias", {"position_bias": 0.8}),
                     ("both", {"length_bias": 1.5, "position_bias": 0.8})]:
        m = point_metrics(simulate(n_questions=400, order_noise=0.6, seed=11, **kw))
        rows.append({"judge": name, "length_bias_pts": round(m["length_bias_pts"], 1),
                     "first_position_win_rate": round(m["first_position_win_rate"], 2),
                     "position_consistency": round(m["position_consistency"], 2),
                     "kappa": round(m["kappa"], 2)})
    return pd.DataFrame(rows)


def coverage(n_reps: int = 300, n_questions: int = 80, pairs_per_question: int = 12) -> pd.DataFrame:
    truth = kappa_stat(simulate(n_questions=20000, pairs_per_question=pairs_per_question, seed=999))
    hits = {"cluster (by question)": 0, "pair (naive)": 0}
    widths = {k: [] for k in hits}
    for r in range(n_reps):
        df = simulate(n_questions=n_questions, pairs_per_question=pairs_per_question, seed=r)
        for name, cl in [("cluster (by question)", "question_id"), ("pair (naive)", None)]:
            lo, hi = kappa_ci(df, n_boot=500, seed=r, cluster=cl)
            hits[name] += lo <= truth <= hi
            widths[name].append(hi - lo)
    return pd.DataFrame([{"interval": k, "coverage_of_90pct_ci": hits[k] / n_reps,
                          "avg_width": float(np.mean(widths[k])), "true_kappa": truth} for k in hits])


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    print("Planted biases vs. what the metrics report\n")
    print(planted_bias_table().to_string(index=False))
    print("\nDoes the 90% interval for kappa cover the true value 90% of the time?\n")
    print(coverage().round(3).to_string(index=False))
