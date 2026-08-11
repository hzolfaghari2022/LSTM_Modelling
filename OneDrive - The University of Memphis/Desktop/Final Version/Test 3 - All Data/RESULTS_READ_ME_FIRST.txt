WHAT THE COMPLETE-DATA TEST SHOWS
=================================

Test data
---------
The complete DC_Offset_147mA experiment was not used for training,
validation, or final fine-tuning. It is therefore a pure unseen test.

The test produced 3,382 consecutive predictions. The first 120 samples
provide the history required by the LSTM, so plotted predictions begin at
0.238 s and continue to the end of the experiment.

Current saved results
---------------------
Displacement:
    RMSE = 0.137255 mm
    MAE  = 0.103215 mm
    R2   = 0.862896, or 86.3% of the measured variation explained
    Fit  = 62.97%

Lorentz force:
    RMSE = 0.00171955 N
    MAE  = 0.00126676 N
    R2   = 0.942534, or 94.3% of the measured variation explained
    Fit  = 76.03%

Simple interpretation
---------------------
1. The Lorentz-force prediction is stronger than the displacement
   prediction.
2. Both predictions follow the overall measured dynamics across the
   untouched test.
3. The largest displacement mismatch occurs around the resonance region,
   approximately 2.4-3.0 s.
4. The model generalizes to the unseen 147 mA experiment, but the
   displacement prediction is not perfect and should not be described as
   100% accurate.

Figures
-------
01_complete_test_overview.png:
    Input, measured outputs, and predictions over the complete test.

02_complete_test_absolute_error.png:
    Where the prediction errors occur over time.

03_complete_test_accuracy_summary.png:
    R2 and fit percentages for both predicted outputs.

To recreate only these figures, run:

    python simple_results_plots.py
