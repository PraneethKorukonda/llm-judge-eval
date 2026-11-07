import pandas as pd

import judgegate.judge as J
from judgegate.data import _split_conversation, majority_human


def _pairs():
    return pd.DataFrame([
        dict(pair_id="1|1|x|y", question_id=1, category="writing", turn=1, model_a="x", model_b="y",
             prompt="q", answer_a="short", answer_b="a much longer answer here"),
        dict(pair_id="2|1|x|y", question_id=2, category="math", turn=1, model_a="x", model_b="y",
             prompt="q", answer_a="long long long long", answer_b="tiny"),
    ])


def test_both_orders_map_back_to_original_labels(tmp_path, monkeypatch):
    calls = []

    def fake(provider, model, prompt, retries=4):
        calls.append(prompt)
        a = prompt.split("[The Start of Assistant A's Answer]\n")[1].split("\n[The End")[0]
        b = prompt.split("[The Start of Assistant B's Answer]\n")[1].split("\n[The End")[0]
        return "[[A]]" if len(a) > len(b) else "[[B]]"   # always prefers the longer answer

    monkeypatch.setattr(J, "call_judge", fake)
    out = J.run_judge(_pairs(), "anthropic", "m", tmp_path / "c.jsonl")
    assert list(out["judge_ab"]) == ["B", "A"]
    assert list(out["judge_ba"]) == ["B", "A"]      # consistent once mapped back
    assert len(calls) == 4

    J.run_judge(_pairs(), "anthropic", "m", tmp_path / "c.jsonl")   # second run hits the cache
    assert len(calls) == 4


def test_first_position_judge_is_inconsistent(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "call_judge", lambda *a, **k: "[[A]]")   # always picks whoever is first
    out = J.run_judge(_pairs(), "anthropic", "m", tmp_path / "c.jsonl")
    assert list(out["judge_ab"]) == ["A", "A"]
    assert list(out["judge_ba"]) == ["B", "B"]


def test_split_conversation_turn_two_has_context():
    conv = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]
    assert _split_conversation(conv, 1) == ("u1", "a1")
    prompt, ans = _split_conversation(conv, 2)
    assert "u1" in prompt and "a1" in prompt and "u2" in prompt and ans == "a2"


def test_majority_vote_and_split_votes_become_tie():
    base = dict(question_id=1, category="writing", turn=1, model_a="x", model_b="y",
                prompt="q", answer_a="a", answer_b="b")
    df = pd.DataFrame([{**base, "pair_id": "p1", "human": h} for h in ["A", "A", "B"]]
                      + [{**base, "pair_id": "p2", "human": h} for h in ["A", "B"]])
    m = majority_human(df).set_index("pair_id")
    assert m.loc["p1", "human"] == "A" and m.loc["p1", "n_raters"] == 3
    assert m.loc["p2", "human"] == "tie"
