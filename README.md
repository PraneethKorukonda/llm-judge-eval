# judgegate: when can an LLM judge replace human raters?

Teams building AI products increasingly use an LLM to grade AI answers instead of paying
human raters. That only works if the LLM judge agrees with people, and doesn't carry
hidden biases like preferring longer answers or whichever answer it reads first.

This project measures an LLM judge against human expert ratings on MT-Bench and then
applies a launch-gate rule per category: **TRUST JUDGE**, **KEEP HUMANS** or
**NEEDS MORE DATA**. It's the same idea as my
[perception release gate](https://github.com/PraneethKorukonda/perception-release-gate),
applied to AI evaluation instead of object detectors.

## What it measures

| Metric | What it tells you |
|---|---|
| Cohen's kappa vs. humans | Agreement beyond chance, with a 90% interval |
| Agreement on decided pairs | When both humans and the judge pick a winner, how often they match |
| Position consistency | Does the judge give the same answer when A and B are swapped? |
| First-position win rate | When it contradicts itself, how often it sides with whichever answer came first (fair = 0.5) |
| Length bias | How much more often the judge picks the longer answer than humans do, in points |

Every pair is judged twice, once in each order. If the two verdicts disagree the final
verdict is a tie, which is the conservative rule from the MT-Bench paper.

Intervals use a **cluster bootstrap by question**. Pairs from the same question are
correlated, so resampling individual pairs gives intervals that are too narrow.

## Checked against known answers first

Before trusting the metrics on a real judge I ran them on a simulated judge with biases I
planted myself (`python scripts/validate_on_simulation.py`, output in `results/simulation_check.txt`):

| Simulated judge | Length bias found | First-position win rate | Position consistency |
|---|---|---|---|
| No bias | 0.0 pts | 0.54 | 0.70 |
| Planted length bias | +16.6 pts | 0.49 | 0.77 |
| Planted position bias | −0.1 pts | 1.00 | 0.46 |
| Both | +13.5 pts | 1.00 | 0.56 |

Each bias shows up where it was planted and nowhere else.

The interval check: over 300 simulated datasets, the naive pair-level 90% interval for kappa
covered the true value only 80% of the time. The question-level interval covered it 86% of
the time. Same lesson as the release gate: resample the unit that's actually independent.

## How to run it

```bash
pip install -r requirements.txt
python -m pytest -q                               # 12 tests, no API calls

export ANTHROPIC_API_KEY=...                      # or OPENAI_API_KEY
python scripts/run_judge_eval.py --n-pairs 400    # about 800 judge calls
python scripts/run_judge_eval.py --n-pairs 400 --provider openai --model gpt-4o-mini
```

The human ratings come from the `lmsys/mt_bench_human_judgments` dataset on Hugging Face
(expert votes from the MT-Bench paper). Question categories come from `data/mt_bench_questions.jsonl`
(copied from the FastChat repo). Judge calls are cached, so re-running is free.

Model names change over time. Pass `--model` with whatever is current, and check pricing before a big run.

Results land in `results/report_<model>.md` and `results/judged_<model>.csv`.

## Results on real data

_Fill in after the run._

| Category | Verdict | Kappa (90% CI) | Position consistency | Length bias |
|---|---|---|---|---|
| ALL | | | | |

Findings:
- 
- 

## Layout

```
judgegate/data.py       load MT-Bench human votes, majority vote per pair
judgegate/judge.py      pairwise judge prompt, both orders, cache, verdict parsing
judgegate/metrics.py    kappa, position and length bias, cluster bootstrap
judgegate/gate.py       per-category TRUST JUDGE / KEEP HUMANS / NEEDS MORE DATA
judgegate/simulate.py   fake judge with planted biases, for validation
scripts/                validation study and the real run
tests/                  12 tests, including a fake judge for the full pipeline
```
