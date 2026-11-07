"""Load MT-Bench human pairwise judgments into one flat table.

Each row is one human vote on one pair of answers to one question turn:
    question_id, category, turn, model_a, model_b, prompt, answer_a, answer_b, human
where `human` is "A", "B" or "tie".

Source: the `lmsys/mt_bench_human_judgments` dataset on Hugging Face (expert votes
from the MT-Bench paper), plus the question file from the FastChat repo for categories.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

QUESTIONS = Path(__file__).resolve().parent.parent / "data" / "mt_bench_questions.jsonl"

_WINNER = {"model_a": "A", "model_b": "B", "tie": "tie", "tie (bothbad)": "tie"}


def load_categories(path: Path = QUESTIONS) -> dict[int, str]:
    with open(path) as f:
        return {q["question_id"]: q["category"] for q in map(json.loads, f)}


def _split_conversation(conv: list[dict], turn: int) -> tuple[str, str]:
    """Return (prompt shown to the judge, answer being judged) for a given turn (1-based).

    For turn 2 the prompt includes the first exchange so the judge sees the context.
    """
    users = [m["content"] for m in conv if m["role"] == "user"]
    bots = [m["content"] for m in conv if m["role"] == "assistant"]
    if turn == 1:
        return users[0], bots[0]
    context = f"User: {users[0]}\nAssistant: {bots[0]}\nUser: {users[1]}"
    return context, bots[1]


def load_human_judgments(split: str = "human") -> pd.DataFrame:
    from datasets import load_dataset  # imported here so tests don't need it

    ds = load_dataset("lmsys/mt_bench_human_judgments", split=split)
    cats = load_categories()
    rows = []
    for r in ds:
        human = _WINNER.get(r["winner"])
        if human is None:
            continue
        prompt, ans_a = _split_conversation(r["conversation_a"], r["turn"])
        _, ans_b = _split_conversation(r["conversation_b"], r["turn"])
        rows.append(dict(
            question_id=r["question_id"], category=cats.get(r["question_id"], "unknown"),
            turn=r["turn"], model_a=r["model_a"], model_b=r["model_b"],
            prompt=prompt, answer_a=ans_a, answer_b=ans_b, human=human, rater=r["judge"],
        ))
    df = pd.DataFrame(rows)
    df["pair_id"] = (df["question_id"].astype(str) + "|" + df["turn"].astype(str) + "|"
                     + df["model_a"] + "|" + df["model_b"])
    return df


def majority_human(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple human votes on the same pair into one label (majority, ties stay ties)."""
    def vote(s: pd.Series) -> str:
        c = s.value_counts()
        top = c[c == c.max()].index.tolist()
        return top[0] if len(top) == 1 else "tie"

    keep = ["pair_id", "question_id", "category", "turn", "model_a", "model_b", "prompt", "answer_a", "answer_b"]
    out = df.groupby(keep, as_index=False).agg(human=("human", vote), n_raters=("human", "size"))
    return out
