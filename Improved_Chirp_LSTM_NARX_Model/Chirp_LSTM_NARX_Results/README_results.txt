Improved Chirp LSTM/NARX Modeling Results
========================================

This model uses past measured outputs and predicts output increments.
This is a one-step-ahead NARX-style system-identification model.

Split mode: within_case
Sequence length: 30 samples = 0.0300 s
Resampling dt: 0.001 s

Generated figures:
Fig01_Training_History: training and validation convergence.
Fig02_*_Prediction: representative true vs predicted trajectories.
Fig03_*_Metric_Summary: RMSE per experiment.
Fig04_*_Parity: predicted-vs-true plots.

Important interpretation:
The default within_case split verifies that the model can learn the measured trajectories.
For harder generalization to unseen loads/cases, rerun with --split leave_load3_out.
