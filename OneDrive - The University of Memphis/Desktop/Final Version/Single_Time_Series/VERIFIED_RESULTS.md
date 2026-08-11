# Verified run - 127 mA record, seed 123, maximum 60 epochs

Validation early stopping ended training after epoch 50 and restored the
best model from epoch 38.

| Output | RMSE | R2 | Fit (%) |
| --- | ---: | ---: | ---: |
| Displacement | 0.10975 mm | 0.91846 | 71.44 |
| Lorentz force | 0.001437 N | 0.94793 | 77.18 |

Window counts: 528 training; 393 validation; 393 test.  The test targets
come from three separated chirp blocks at 1.5-2.0, 4.0-4.5, and
6.5-7.0 seconds.

Small numerical differences are expected across PyTorch versions and
hardware.  `ResultsData` and `FiguresResults` contain the verified run.
