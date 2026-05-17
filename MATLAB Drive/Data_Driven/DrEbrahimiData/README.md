# LSTM Excel Data Training MATLAB

This repository contains a MATLAB based workflow for reading Excel datasets, analyzing sheet information, preprocessing numerical data, and training an LSTM network for data driven modeling and time series prediction.

## Main Features

- Reads Excel files with multiple sheets
- Displays sheet names, row counts, column counts, and variable names
- Extracts numeric columns automatically
- Checks missing values
- Removes incomplete rows
- Normalizes data for neural network training
- Prepares sequence data for LSTM training
- Trains an LSTM regression network
- Saves prediction plots and processed MATLAB data
- Pushes code, data, and results to GitHub

## Excel File

The current Excel file used in this project is:

`data- 5-26.xlsx`

## Main MATLAB File

`Main_Read_Excel_Train_LSTM.m`

## Notes

By default, the code uses the first sheet in the Excel file for LSTM training. The last numeric column is considered the output, and all previous numeric columns are considered input features. This can be changed inside the MATLAB code depending on the structure of the dataset.
