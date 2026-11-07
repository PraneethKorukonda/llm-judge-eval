"""Run an LLM judge on MT-Bench pairs that humans already rated, then apply the trust gate.

Usage:
    export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY with --provider openai
    python scripts/run_judge_eval.py --n-pairs 400
    python scripts/run_judge_eval.py --n-pairs 400 --provider openai --model gpt-4o-mini

Outputs (in results/):
    judged_<model>.csv   one row per pair with human and judge verdicts
    report_<model>.md    gate verdicts and metrics by category
Re-running is cheap: every judge call is cached in results/cache_<model>.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from judgegate.data import load_human_judgments, majority_human  # noqa: E402
from judgegate.gate import GateRules, evaluate  # noqa: E402
from judgegate.judge import default_model, run_judge  # noqa: E402
from judgegate.metrics import add_lengths, kappa_ci  # noqa: E402


def sample_pairs(pairs: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Same number of pairs from each category, so no category is starved."""
    per_cat = max(1, n // pairs["category"].nunique())
    return pairs.sample(frac=1, random_state=seed).groupby("category").head(per_cat).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--turn", type=int, default=1, choices=[1, 2])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model = args.model or default_model(args.provider)

    votes = load_human_judgments()
    print(f"{len(votes)} human votes on {votes['pair_id'].nunique()} pairs")
    pairs = majority_human(votes[votes["turn"] == args.turn])
    pairs = sample_pairs(pairs, args.n_pairs, args.seed)
    print(f"judging {len(pairs)} pairs from {pairs['question_id'].nunique()} questions with {model}")

    out = ROOT / "results"
    tag = model.replace("/", "_")
    judged = run_judge(pairs, args.provider, model, out / f"cache_{tag}.jsonl")
    judged = add_lengths(judged)
    unparsed = judged[["judge_ab", "judge_ba"]].isna().any(axis=1).sum()
    judged = judged.dropna(subset=["judge_ab", "judge_ba"])
    judged.drop(columns=["prompt", "answer_a", "answer_b"]).to_csv(out / f"judged_{tag}.csv", index=False)

    table = evaluate(judged, GateRules())
    c_lo, c_hi = kappa_ci(judged)
    p_lo, p_hi = kappa_ci(judged, cluster=None)
    cols = ["category", "verdict", "n_questions", "n_pairs", "kappa", "kappa_lo", "kappa_hi",
            "agreement_decided", "position_consistency", "first_position_win_rate", "length_bias_pts"]
    report = [
        f"# LLM judge vs. human raters: {model}\n",
        f"{len(judged)} MT-Bench pairs (turn {args.turn}), {unparsed} dropped for unparseable verdicts.\n",
        "## Gate verdicts by category\n",
        table[cols].round(3).to_markdown(index=False),
        "\n## Interval check\n",
        f"Kappa 90% interval, resampling questions: {c_lo:.3f} to {c_hi:.3f}  ",
        f"Kappa 90% interval, resampling pairs (naive): {p_lo:.3f} to {p_hi:.3f}\n",
    ]
    (out / f"report_{tag}.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
