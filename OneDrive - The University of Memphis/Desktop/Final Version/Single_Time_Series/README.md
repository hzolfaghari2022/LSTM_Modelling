# FARNN Python - one selectable chirp time series

This independent codebase demonstrates identification using only one of
the five workbook records.  It uses the same model as the five-series
version:

`1 measured current -> 3 features -> 32 -> 64 -> 64 -> 2 outputs`

The selected record is divided into repeating 0.5-second blocks with the
pattern `train, validation, train, test, train`.  Consequently, training,
validation, and test targets occur at several different times/frequencies
instead of each occupying one continuous part of the chirp.  Every LSTM
window stays inside one role block.

## Choose the record

The default is `DC_Offset_127mA`.  To change it permanently, edit this line
in `config.py`:

```python
SINGLE_SERIES_SHEET = os.environ.get(
    "FARNN_SINGLE_SHEET",
    "DC_Offset_127mA",
)
```

Or choose any record temporarily in PowerShell:

```powershell
$env:FARNN_SINGLE_SHEET="DC_Offset_87mA"
python main.py
```

Allowed names are `DC_Offset_67mA`, `DC_Offset_87mA`,
`DC_Offset_107mA`, `DC_Offset_127mA`, and `DC_Offset_147mA`.

## Run

```powershell
conda activate lstm-py312
python main.py
```

The model, split table, metrics, and predictions are saved in
`ResultsData`; figures are saved in `FiguresResults`.

The default maximum is 60 epochs, with validation early stopping.  On the
included 127 mA record and seed 123, the verified run stopped after epoch
50 and restored epoch 38.

Reference repository: https://github.com/robotsorcerer/FARNN
