"""Synthetic judge data with known answers, used to check the metrics before trusting them.

Each question has a difficulty effect shared by all its pairs (so pairs are correlated
within a question, like real data). Humans vote on the true quality gap. The fake judge
sees the same gap plus noise, plus a planted length bias and a planted position bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]


def simulate(n_questions: int = 80, pairs_per_question: int = 12, judge_noise: float = 0.6,
             length_bias: float = 0.0, position_bias: float = 0.0, order_noise: float = 0.2,
             tie_band: float = 0.35, q_hardness: float = 0.8,
             seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_questions):
        cat = CATEGORIES[q % len(CATEGORIES)]
        q_shift = rng.normal(0, 0.8)            # shared within question -> clustering
        # Some questions are much harder for the judge than others (shared by all their pairs).
        q_noise = judge_noise * (1.6 if cat in ("math", "reasoning") else 1.0) * rng.lognormal(0, q_hardness)
        for k in range(pairs_per_question):
            gap = rng.normal(0, 1) + 0.3 * q_shift   # true quality of A minus B
            len_a, len_b = rng.integers(60, 600, 2)
            ldiff = (len_a - len_b) / 300.0

            def vote(score: float) -> str:
                return "tie" if abs(score) < tie_band else ("A" if score > 0 else "B")

            human = vote(gap + rng.normal(0, 0.3))
            shared = gap + length_bias * ldiff + rng.normal(0, q_noise) + 0.5 * q_shift * q_noise
            ab = vote(shared + position_bias + rng.normal(0, order_noise))      # A shown first
            ba = vote(shared - position_bias + rng.normal(0, order_noise))      # B shown first
            rows.append(dict(pair_id=f"{q}|{k}", question_id=q, category=cat, turn=1,
                             human=human, judge_ab=ab, judge_ba=ba, len_a=int(len_a), len_b=int(len_b)))
    return pd.DataFrame(rows)
