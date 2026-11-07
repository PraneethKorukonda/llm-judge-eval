"""Run an LLM as a pairwise judge, in both answer orders, with an on-disk cache.

Running both orders is what lets us measure position bias: a fair judge should give
the same winner when A and B are swapped.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd

# Adapted from the pairwise prompt in the MT-Bench paper (Zheng et al., 2023).
SYSTEM = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
    "assistants to the user question displayed below. Choose the assistant that follows the user's "
    "instructions and answers the user's question better. Consider helpfulness, relevance, accuracy, "
    "depth, creativity and level of detail. Do not let the order of the responses, their length or the "
    "names of the assistants influence your decision. Be as objective as possible. After a short "
    'explanation, output your final verdict strictly in this format: "[[A]]" if assistant A is better, '
    '"[[B]]" if assistant B is better, and "[[C]]" for a tie.'
)

TEMPLATE = (
    "[User Question]\n{prompt}\n\n"
    "[The Start of Assistant A's Answer]\n{a}\n[The End of Assistant A's Answer]\n\n"
    "[The Start of Assistant B's Answer]\n{b}\n[The End of Assistant B's Answer]"
)

_VERDICT = re.compile(r"\[\[(A|B|C)\]\]")


def parse_verdict(text: str) -> str | None:
    hits = _VERDICT.findall(text or "")
    if not hits:
        return None
    return {"A": "A", "B": "B", "C": "tie"}[hits[-1]]


def _call_anthropic(model: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(model=model, max_tokens=600, temperature=0, system=SYSTEM,
                                 messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def _call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=600,
                                       messages=[{"role": "system", "content": SYSTEM},
                                                 {"role": "user", "content": prompt}])
    return r.choices[0].message.content


def call_judge(provider: str, model: str, prompt: str, retries: int = 4) -> str:
    fn = {"anthropic": _call_anthropic, "openai": _call_openai}[provider]
    for i in range(retries):
        try:
            return fn(model, prompt)
        except Exception as e:  # rate limits, timeouts
            if i == retries - 1:
                raise
            time.sleep(2 ** i * 5)
            print(f"retry {i + 1} after error: {e}")
    raise RuntimeError("unreachable")


def run_judge(pairs: pd.DataFrame, provider: str, model: str, cache: Path) -> pd.DataFrame:
    """Judge every pair in both orders. Returns one row per pair with judge_ab and judge_ba.

    judge_ab: verdict with the original order (A shown first).
    judge_ba: verdict with answers swapped, mapped back to the original labels.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache.exists():
        with open(cache) as f:
            for line in f:
                r = json.loads(line)
                done[(r["pair_id"], r["order"])] = r

    flip = {"A": "B", "B": "A", "tie": "tie", None: None}
    with open(cache, "a") as out:
        for i, p in enumerate(pairs.itertuples(index=False)):
            for order in ("ab", "ba"):
                key = (p.pair_id, order)
                if key in done:
                    continue
                a, b = (p.answer_a, p.answer_b) if order == "ab" else (p.answer_b, p.answer_a)
                text = call_judge(provider, model, TEMPLATE.format(prompt=p.prompt, a=a, b=b))
                v = parse_verdict(text)
                rec = {"pair_id": p.pair_id, "order": order, "raw": text,
                       "verdict": v if order == "ab" else flip[v]}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done[key] = rec
            if (i + 1) % 25 == 0:
                print(f"judged {i + 1}/{len(pairs)} pairs")

    res = pairs.copy()
    res["judge_ab"] = [done[(pid, "ab")]["verdict"] for pid in res["pair_id"]]
    res["judge_ba"] = [done[(pid, "ba")]["verdict"] for pid in res["pair_id"]]
    return res


def default_model(provider: str) -> str:
    return os.environ.get("JUDGE_MODEL") or {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}[provider]
