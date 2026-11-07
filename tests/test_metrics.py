import numpy as np
import pandas as pd

from judgegate.gate import GateRules, evaluate
from judgegate.judge import parse_verdict
from judgegate.metrics import cohen_kappa, combined_verdict, kappa_ci, point_metrics
from judgegate.simulate import simulate


def test_parse_verdict_takes_last_tag():
    assert parse_verdict("A is good... final: [[B]]") == "B"
    assert parse_verdict("[[A]] wait, actually [[C]]") == "tie"
    assert parse_verdict("no verdict here") is None


def test_inconsistent_orders_become_tie():
    v = combined_verdict(pd.Series(["A", "A", "B"]), pd.Series(["A", "B", "B"]))
    assert list(v) == ["A", "tie", "B"]


def test_kappa_known_values():
    x = pd.Series(["A", "B", "tie", "A"])
    assert cohen_kappa(x, x) == 1.0
    assert abs(cohen_kappa(pd.Series(["A", "B"] * 50), pd.Series(["A"] * 50 + ["B"] * 50))) < 0.05


def test_no_planted_bias_reads_as_no_bias():
    m = point_metrics(simulate(n_questions=200, order_noise=1.0, seed=1))
    assert abs(m["length_bias_pts"]) < 4
    assert abs(m["first_position_win_rate"] - 0.5) < 0.15


def test_recovers_planted_length_bias():
    m = point_metrics(simulate(n_questions=200, length_bias=1.5, seed=2))
    assert m["length_bias_pts"] > 8


def test_recovers_planted_position_bias():
    m = point_metrics(simulate(n_questions=200, position_bias=0.8, seed=3))
    assert m["first_position_win_rate"] > 0.85
    assert m["position_consistency"] < point_metrics(simulate(n_questions=200, seed=3))["position_consistency"]


def test_cluster_interval_is_wider_than_pair_interval():
    df = simulate(n_questions=60, pairs_per_question=15, seed=4)
    c_lo, c_hi = kappa_ci(df, n_boot=800)
    p_lo, p_hi = kappa_ci(df, n_boot=800, cluster=None)
    assert (c_hi - c_lo) > 1.2 * (p_hi - p_lo)


def test_gate_verdicts():
    rules = GateRules(n_boot=400)
    good = evaluate(simulate(n_questions=160, judge_noise=0.15, q_hardness=0.0, seed=5), rules)
    assert good.loc[good.category == "ALL", "verdict"].item() == "TRUST JUDGE"
    bad = evaluate(simulate(n_questions=160, judge_noise=2.0, seed=6), rules)
    assert bad.loc[bad.category == "ALL", "verdict"].item() == "KEEP HUMANS"
    tiny = evaluate(simulate(n_questions=16, seed=7), rules)
    assert tiny.loc[tiny.category == "math", "verdict"].item() == "NEEDS MORE DATA"
