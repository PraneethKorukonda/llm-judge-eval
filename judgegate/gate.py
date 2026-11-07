"""Decide, per category, whether the LLM judge can replace human raters.

Same idea as a model release gate: set the bar in advance, look at the lower end of
the interval, and refuse to decide when there isn't enough data.

    TRUST JUDGE      kappa's lower 90% bound clears the bar and the judge is position-consistent
    KEEP HUMANS      enough data, but the judge doesn't clear the bar
    NEEDS MORE DATA  too few questions to tell
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import kappa_ci, point_metrics


@dataclass(frozen=True)
class GateRules:
    kappa_floor: float = 0.60          # "substantial" agreement on the Landis-Koch scale
    min_position_consistency: float = 0.75
    min_questions: int = 8
    n_boot: int = 1000


def evaluate(df: pd.DataFrame, rules: GateRules = GateRules(), by: str = "category") -> pd.DataFrame:
    rows = []
    for name, g in [("ALL", df)] + list(df.groupby(by)):
        m = point_metrics(g)
        lo, hi = kappa_ci(g, n_boot=rules.n_boot)
        if m["n_questions"] < rules.min_questions:
            verdict = "NEEDS MORE DATA"
        elif lo >= rules.kappa_floor and m["position_consistency"] >= rules.min_position_consistency:
            verdict = "TRUST JUDGE"
        else:
            verdict = "KEEP HUMANS"
        rows.append({by: name, **m, "kappa_lo": lo, "kappa_hi": hi, "verdict": verdict})
    return pd.DataFrame(rows)
