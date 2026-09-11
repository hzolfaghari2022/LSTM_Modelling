# V15 cumulative 1-to-13 feature comparison

## What was tested

Thirteen independently trained models used the same random seed, data split, 64-sample history, two 48-unit LSTM layers, and 48-to-32-to-1 displacement-residual head. F01 uses current only. Each following case adds exactly one causal feature until F13 uses all 13.

The ablation affects only the LSTM displacement correction. Lorentz force is produced by the unchanged separate causal force rule, so force accuracy is not evidence for or against an LSTM feature.

## Validation-based comparison

The lowest pooled validation displacement RMSE occurred for F13 (13 features): 0.00024204039 mm, with fit 99.9844%. This is the defensible ranking because validation data—not pure-test data—must guide model selection.

## Pure-test confirmation

Across the 13 cases, pooled pure-test displacement RMSE ranged from 0.00026362925 to 0.00052513094 mm. The maximum pointwise prediction difference between a reduced model and F13 was 0.0045537648 mm.

If the curves still look visually similar, inspect `03_feature_effect_audit.png` and `prediction_difference_vs_F13.csv`. The original quadratic displacement-history baseline is already strong, so current-only LSTM corrections can still produce high one-step tracking. Results were not artificially degraded to force a preferred conclusion.

## Interpretation limits

- More features do not automatically improve generalization; redundant or noisy features can leave accuracy unchanged or make validation performance worse.
- This is cumulative, order-dependent ablation. It answers what happens as features are added in the stated order; it is not a unique importance ranking for every feature.
- These are one-step measured-feedback results, not autonomous/free-running simulation results.
- Do not select a feature count from pure-test performance. Use the validation ranking, then describe pure-test results as confirmation.
