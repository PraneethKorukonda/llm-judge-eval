"""How much can the LLM judge be trusted, compared with human raters?

All confidence intervals use a cluster bootstrap by question: pairs from the same
question are correlated, so resampling individual pairs makes intervals too narrow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["A", "B", "tie"]


def combined_verdict(ab: pd.Series, ba: pd.Series) -> pd.Series:
    """Final judge verdict: keep it if both orders agree, otherwise call it a tie
    (the MT-Bench paper's conservative rule)."""
    return pd.Series(np.where(ab == ba, ab, "tie"), index=ab.index)


def cohen_kappa(x: pd.Series, y: pd.Series, labels=LABELS) -> float:
    x, y = np.asarray(x), np.asarray(y)
    po = np.mean(x == y)
    pe = sum(np.mean(x == l) * np.mean(y == l) for l in labels)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def point_metrics(df: pd.DataFrame) -> dict:
    """df needs: human, judge_ab, judge_ba, len_a, len_b."""
    judge = combined_verdict(df["judge_ab"], df["judge_ba"])
    decided = df["human"] != "tie"
    jd = judge != "tie"

    consistent = df["judge_ab"] == df["judge_ba"]
    flipped = ~consistent & df["judge_ab"].isin(["A", "B"]) & df["judge_ba"].isin(["A", "B"])
    # When the judge contradicts itself, does it side with whichever answer was shown first?
    # In the ab run the first answer is A; in the ba run the first answer is B (mapped back).
    first_wins = ((df["judge_ab"] == "A") & (df["judge_ba"] == "B"))
    position_bias = float(first_wins[flipped].mean()) if flipped.any() else float("nan")

    # Length bias: among pairs where humans picked a winner and the judge also did,
    # how often each side picked the longer answer.
    longer = np.where(df["len_a"] >= df["len_b"], "A", "B")
    both = decided & jd
    judge_longer = float((judge[both] == longer[both]).mean()) if both.any() else float("nan")
    human_longer = float((df.loc[both, "human"] == longer[both]).mean()) if both.any() else float("nan")

    return {
        "n_pairs": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "agreement_3way": float((judge == df["human"]).mean()),
        "agreement_decided": float((judge[decided & jd] == df.loc[decided & jd, "human"]).mean())
        if (decided & jd).any() else float("nan"),
        "kappa": cohen_kappa(df["human"], judge),
        "position_consistency": float(consistent.mean()),
        "first_position_win_rate": position_bias,
        "judge_picks_longer": judge_longer,
        "human_picks_longer": human_longer,
        "length_bias_pts": (judge_longer - human_longer) * 100,
    }


def cluster_bootstrap(df: pd.DataFrame, stat, n_boot: int = 2000, seed: int = 0,
                      cluster: str = "question_id") -> tuple[float, float]:
    """90% interval for stat(df), resampling whole questions."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        vals.append(stat(pd.concat([groups[k] for k in pick], ignore_index=True)))
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def pair_bootstrap(df: pd.DataFrame, stat, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Naive 90% interval that resamples pairs. Kept only to show how much too narrow it is."""
    rng = np.random.default_rng(seed)
    n = len(df)
    vals = np.array([stat(df.iloc[rng.integers(0, n, n)]) for _ in range(n_boot)], dtype=float)
    vals = vals[~np.isnan(vals)]
    return float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95))


def _confusions(df: pd.DataFrame, cluster: str) -> np.ndarray:
    """One 3x3 human-by-judge count matrix per cluster."""
    idx = {l: i for i, l in enumerate(LABELS)}
    judge = combined_verdict(df["judge_ab"], df["judge_ba"])
    h = df["human"].map(idx).to_numpy()
    j = judge.map(idx).to_numpy()
    codes, uniq = pd.factorize(df[cluster])
    out = np.zeros((len(uniq), 3, 3))
    np.add.at(out, (codes, h, j), 1)
    return out


def _kappa_from_counts(c: np.ndarray) -> float:
    n = c.sum()
    po = np.trace(c) / n
    pe = float((c.sum(1) / n) @ (c.sum(0) / n))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kappa_ci(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0, cluster: str | None = "question_id",
             level: float = 0.90) -> tuple[float, float]:
    """Fast bootstrap interval for kappa. cluster=None resamples individual pairs (naive)."""
    rng = np.random.default_rng(seed)
    if cluster is None:
        df = df.assign(_row=np.arange(len(df)))
        cluster = "_row"
    conf = _confusions(df, cluster)
    k = len(conf)
    picks = rng.integers(0, k, size=(n_boot, k))
    weights = np.apply_along_axis(np.bincount, 1, picks, minlength=k)
    totals = np.einsum("bk,kij->bij", weights, conf)
    vals = np.array([_kappa_from_counts(t) for t in totals])
    a = (1 - level) / 2
    return float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))


def kappa_stat(d: pd.DataFrame) -> float:
    return cohen_kappa(d["human"], combined_verdict(d["judge_ab"], d["judge_ba"]))


def add_lengths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["len_a"] = df["answer_a"].str.split().str.len()
    df["len_b"] = df["answer_b"].str.split().str.len()
    return df
