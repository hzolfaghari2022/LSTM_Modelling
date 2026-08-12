# Verified run - included workbook, seed 123, 20 epochs

| Evaluation | Output | RMSE | R2 | Fit (%) |
| --- | --- | ---: | ---: | ---: |
| 127 mA distributed internal test | Displacement | 0.07019 mm | 0.96665 | 81.74 |
| 127 mA distributed internal test | Lorentz force | 0.001311 N | 0.95667 | 79.18 |
| 147 mA untouched pure test | Displacement | 0.11072 mm | 0.91078 | 70.13 |
| 147 mA untouched pure test | Lorentz force | 0.002084 N | 0.91561 | 70.95 |

Window counts: 5,601 training; 393 validation; 393 fourth-series
internal test; 3,382 fifth-series pure test.

Small numerical differences are expected across PyTorch versions and
hardware.  `ResultsData` and `FiguresResults` contain the verified run.

