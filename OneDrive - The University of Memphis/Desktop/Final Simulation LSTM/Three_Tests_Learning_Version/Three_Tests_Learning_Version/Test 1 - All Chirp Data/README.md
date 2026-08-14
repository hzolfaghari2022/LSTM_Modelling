# Deep LSTM system identification - five chirp time series

For a block-by-block explanation, open `../CODE_WALKTHROUGH.md`.

This folder keeps the paper's main modeling pattern: three stacked LSTM
modules, dropout after each LSTM, and a final linear output layer.  The
necessary plant-specific dimensions are:

`1 measured current -> 3 features -> 32 -> 64 -> 64 -> 2 outputs`

The three input features are current, current change, and DC operating
point.  The two outputs are displacement and Lorentz force.

## Required data roles

- `DC_Offset_67mA`, `DC_Offset_87mA`, `DC_Offset_107mA`: complete records
  used only for training.
- `DC_Offset_127mA`: repeating 0.5-second training, validation, and test
  blocks spread over the entire chirp.
- `DC_Offset_147mA`: complete untouched pure test.

The repeating role pattern on the fourth record is
`train, validation, train, test, train`, giving an approximate 60/20/20
split.  Every LSTM window remains inside one block, so targets from one
role cannot leak into another role.

## Run

```powershell
conda activate lstm-py312
python main.py
```

`main.py` trains, evaluates both tests, saves the model and CSV results in
`ResultsData`, and creates figures in `FiguresResults`.

For a temporary 30-epoch run in PowerShell:

```powershell
$env:DLSTM_EPOCHS="30"
python main.py
```

## Alignment with the reference

The paper and original Lua/Torch repository use stacked recurrent modules,
dropout, a final linear layer, MSE-based gradient training, mini-batches,
and seed 123.  This Python version retains those concepts.  Changes that
are necessary for this plant are the 3 input features, hidden sizes
32/64/64, 2 outputs, COMSOL workbook reader, anti-aliased downsampling,
sequence windows, validation-controlled training, and the five-record
partition used in this study.
