# LLM judge vs. human raters: claude-haiku-5-5

386 MT-Bench pairs (turn 1), 14 dropped for unparseable verdicts.

## Gate verdicts by category

| category   | verdict     |   n_questions |   n_pairs |   kappa |   kappa_lo |   kappa_hi |   agreement_decided |   position_consistency |   first_position_win_rate |   length_bias_pts |
|:-----------|:------------|--------------:|----------:|--------:|-----------:|-----------:|--------------------:|-----------------------:|--------------------------:|------------------:|
| ALL        | KEEP HUMANS |            79 |       386 |   0.477 |      0.417 |      0.538 |               0.88  |                  0.894 |                     0.625 |            -0.8   |
| coding     | KEEP HUMANS |            10 |        46 |   0.449 |      0.246 |      0.649 |               0.828 |                  0.891 |                     0     |            10.345 |
| extraction | KEEP HUMANS |             9 |        47 |   0.588 |      0.339 |      0.789 |               0.969 |                  0.936 |                   nan     |            -3.125 |
| humanities | KEEP HUMANS |            10 |        49 |   0.307 |      0.195 |      0.459 |               0.794 |                  0.98  |                     1     |             2.941 |
| math       | TRUST JUDGE |            10 |        50 |   0.756 |      0.65  |      0.848 |               0.943 |                  0.86  |                   nan     |             0     |
| reasoning  | KEEP HUMANS |            10 |        47 |   0.426 |      0.196 |      0.596 |               1     |                  0.851 |                   nan     |             0     |
| roleplay   | KEEP HUMANS |            10 |        50 |   0.274 |      0.102 |      0.464 |               0.757 |                  0.92  |                     0.667 |           -13.514 |
| stem       | KEEP HUMANS |            10 |        49 |   0.49  |      0.304 |      0.642 |               0.886 |                  0.939 |                     0     |             0     |
| writing    | KEEP HUMANS |            10 |        48 |   0.452 |      0.35  |      0.532 |               0.933 |                  0.771 |                     0.7   |             0     |

## Interval check

Kappa 90% interval, resampling questions: 0.415 to 0.542  
Kappa 90% interval, resampling pairs (naive): 0.418 to 0.533
