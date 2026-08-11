FARNN ACCURACY-IMPROVED VERIFIED VERSION
========================================

RUN ONLY
--------
    conda activate lstm-py312
    python main.py

main.py trains the model, evaluates the untouched 147 mA record, saves
all numerical results, creates all figures, and opens FiguresResults on
Windows.

MODEL
-----
The model stays simple:

    one measured current
    -> [I, delta_I, I_DC]
    -> LSTM 32
    -> LSTM 64
    -> LSTM 64
    -> Linear 2

Outputs:
    displacement
    Lorentz force

MAIN TECHNICAL FIX
------------------
The original data are sampled at 2000 Hz, while the chirp ends at 31 Hz.
The code now uses scipy.signal.resample_poly to perform anti-aliased
reduction by a factor of four:

    2000 Hz -> 500 Hz

This keeps the physical chirp information while making a 120-sample LSTM
window represent 0.24 s instead of only 0.06 s. It also reduces training
cost substantially.

OTHER TARGETED CHANGES
----------------------
1. Input features remain [I, delta_I, I_DC].
2. Dropout is reduced from 0.30 to 0.10.
3. Adam learning-rate reduction is validation-controlled.
4. Early stopping restores the best validation epoch.
5. Large displacement responses receive moderate extra importance.
6. Measured curves are green solid lines.
7. Predicted curves are orange dashed lines.
8. main.py runs plotting automatically.

DATA
----
67, 87, 107, and 127 mA:
    distributed training and validation blocks

147 mA:
    complete untouched pure final test

CURRENT SAVED RUN
-----------------
The included ResultsData, FiguresResults, and SimpleResultsFigures were
generated with the included workbook and random seed 123. The numerical
values below come directly from the current ResultsData/metrics.csv file.

Pure 147 mA test:

    Displacement:
        RMSE = 0.13725 mm
        R2 = 0.86290
        Fit = 62.97%

    Lorentz force:
        RMSE = 0.001720 N
        R2 = 0.94253
        Fit = 76.03%

Small numerical differences can occur on another computer or PyTorch
version.

OPTIONAL LONGER RUN
-------------------
To try 30 epochs in PowerShell:

    $env:FARNN_EPOCHS="30"
    python main.py

The 147 mA record is outside the development current range, so zero test
error cannot be guaranteed honestly.


SIMPLE RESULTS FIGURES
----------------------
Run:

    python simple_results_plots.py

The separate plotting code creates three easy-to-read figures in
SimpleResultsFigures without retraining or changing the model.


MINIMAL FINAL ACCURACY IMPROVEMENT
----------------------------------
No layer was added.

After the best validation epoch is restored, the same model is
fine-tuned for four epochs using all training and validation windows
from 67, 87, 107, and 127 mA.

Fine-tuning learning rate:
    0.0001

The complete 147 mA record remains untouched until the final test.

This is smaller and safer than adding another LSTM layer because the
current loss curve was still improving and the main limitation was
generalization from the development records to 147 mA.
