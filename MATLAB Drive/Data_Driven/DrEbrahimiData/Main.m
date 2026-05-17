clc;
clear;
close all;

%% ============================================================
%  Main_Read_Excel_Train_LSTM.m
%
%  Purpose:
%  This code reads an Excel file, analyzes all sheets, extracts
%  numeric data, prepares the data for LSTM training, trains an
%  example LSTM regression model, saves the results, and pushes
%  the files to GitHub.
%
%  Excel file:
%  data- 5-26.xlsx
%
%  GitHub repo:
%  https://github.com/hzolfaghari2022/LSTM_Excel_Data_Training_MATLAB.git
%% ============================================================

%% User Settings

fileName = 'data- 5-26.xlsx';

repoURL = 'https://github.com/hzolfaghari2022/LSTM_Excel_Data_Training_MATLAB.git';

resultsFolder = 'Results_LSTM_Excel_Data';

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

%% Check Excel File

if ~isfile(fileName)
    error('The Excel file "%s" was not found in the current MATLAB folder.', fileName);
end

fprintf('\nExcel file found: %s\n', fileName);

%% Get Sheet Names

sheetList = sheetnames(fileName);

fprintf('\nAvailable sheets in the Excel file:\n');
disp(sheetList);

%% Read and Analyze All Sheets

allSheetsData = struct();
sheetSummaryTable = table();

for i = 1:numel(sheetList)

    sheetName = sheetList{i};
    validSheetName = matlab.lang.makeValidName(sheetName);

    fprintf('\n====================================================\n');
    fprintf('Reading sheet: %s\n', sheetName);
    fprintf('====================================================\n');

    dataTable = readtable(fileName, 'Sheet', sheetName);

    allSheetsData.(validSheetName) = dataTable;

    numRows = height(dataTable);
    numColumns = width(dataTable);
    variableNames = dataTable.Properties.VariableNames;

    fprintf('Number of rows: %d\n', numRows);
    fprintf('Number of columns: %d\n', numColumns);

    fprintf('\nColumn names:\n');
    disp(variableNames');

    fprintf('\nFirst few rows:\n');
    disp(head(dataTable));

    fprintf('\nData summary:\n');
    summary(dataTable);

    newRow = table( ...
        string(sheetName), ...
        numRows, ...
        numColumns, ...
        string(strjoin(variableNames, ', ')), ...
        'VariableNames', {'SheetName', 'NumberOfRows', 'NumberOfColumns', 'ColumnNames'} ...
    );

    sheetSummaryTable = [sheetSummaryTable; newRow];

end

%% Save Excel Sheet Information

summaryFileName = fullfile(resultsFolder, 'Excel_Sheet_Information.xlsx');
writetable(sheetSummaryTable, summaryFileName);

fprintf('\nSheet information saved to:\n%s\n', summaryFileName);

%% Select Main Sheet for LSTM

% By default, this code uses the first sheet.
% If your important data is in another sheet, change this line.
mainSheetName = sheetList{1};

fprintf('\nMain sheet selected for LSTM training: %s\n', mainSheetName);

dataTable = readtable(fileName, 'Sheet', mainSheetName);

%% Extract Numeric Data Only

numericDataTable = dataTable(:, vartype('numeric'));

if width(numericDataTable) < 2
    error('The selected sheet must contain at least two numeric columns for LSTM training.');
end

fprintf('\nNumeric columns selected for LSTM:\n');
disp(numericDataTable.Properties.VariableNames');

dataMatrixOriginal = table2array(numericDataTable);

fprintf('\nOriginal numeric data size:\n');
fprintf('Rows: %d\n', size(dataMatrixOriginal, 1));
fprintf('Columns: %d\n', size(dataMatrixOriginal, 2));

%% Check Missing Values

missingValuesPerColumn = sum(ismissing(numericDataTable));

fprintf('\nMissing values in each numeric column:\n');
disp(array2table(missingValuesPerColumn, ...
    'VariableNames', numericDataTable.Properties.VariableNames));

%% Remove Missing Rows

dataMatrix = rmmissing(dataMatrixOriginal);

fprintf('\nNumeric data size after removing missing rows:\n');
fprintf('Rows: %d\n', size(dataMatrix, 1));
fprintf('Columns: %d\n', size(dataMatrix, 2));

%% Plot Raw Numeric Data

figure('Color', 'w', 'Name', 'Raw Numeric Data');
plot(dataMatrix, 'LineWidth', 1.5);
grid on;
xlabel('Sample');
ylabel('Value');
title('Raw Numeric Data from Excel');
legend(numericDataTable.Properties.VariableNames, 'Interpreter', 'none', 'Location', 'best');

saveas(gcf, fullfile(resultsFolder, 'Raw_Numeric_Data.png'));
savefig(gcf, fullfile(resultsFolder, 'Raw_Numeric_Data.fig'));

%% Normalize Data

dataMean = mean(dataMatrix, 1);
dataStd = std(dataMatrix, 0, 1);

% Avoid division by zero for constant columns
dataStd(dataStd == 0) = 1;

dataNormalized = (dataMatrix - dataMean) ./ dataStd;

%% Plot Normalized Data

figure('Color', 'w', 'Name', 'Normalized Numeric Data');
plot(dataNormalized, 'LineWidth', 1.5);
grid on;
xlabel('Sample');
ylabel('Normalized Value');
title('Normalized Numeric Data for LSTM');
legend(numericDataTable.Properties.VariableNames, 'Interpreter', 'none', 'Location', 'best');

saveas(gcf, fullfile(resultsFolder, 'Normalized_Numeric_Data.png'));
savefig(gcf, fullfile(resultsFolder, 'Normalized_Numeric_Data.fig'));

%% Prepare Data for LSTM

% Assumption:
% The last numeric column is the output.
% All previous numeric columns are inputs.
%
% Example:
% Column 1 to column end minus 1  --> input features
% Last column                    --> target output

inputData = dataNormalized(:, 1:end-1);
outputData = dataNormalized(:, end);

numFeatures = size(inputData, 2);
numResponses = 1;

fprintf('\nLSTM data preparation:\n');
fprintf('Number of input features: %d\n', numFeatures);
fprintf('Number of output responses: %d\n', numResponses);

%% Convert Data to Sequence Format

% LSTM in MATLAB expects sequence data in the format:
% features by time steps

X = inputData';
Y = outputData';

%% Split Data into Training and Testing

numSamples = size(X, 2);

trainRatio = 0.8;
numTrainSamples = floor(trainRatio * numSamples);

XTrain = {X(:, 1:numTrainSamples)};
YTrain = {Y(:, 1:numTrainSamples)};

XTest = {X(:, numTrainSamples+1:end)};
YTest = {Y(:, numTrainSamples+1:end)};

fprintf('\nTraining samples: %d\n', numTrainSamples);
fprintf('Testing samples: %d\n', numSamples - numTrainSamples);

%% Define LSTM Network

numHiddenUnits = 100;

layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input')

    lstmLayer(numHiddenUnits, ...
    'OutputMode', 'sequence', ...
    'Name', 'lstm')

    fullyConnectedLayer(numResponses, 'Name', 'fully_connected')

    regressionLayer('Name', 'regression_output')
];

%% Training Options

options = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'MiniBatchSize', 1, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 100, ...
    'LearnRateDropFactor', 0.5, ...
    'Shuffle', 'never', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% Train LSTM Network

fprintf('\nTraining LSTM network...\n');

net = trainNetwork(XTrain, YTrain, layers, options);

fprintf('LSTM training completed.\n');

%% Predict on Test Data

YPred = predict(net, XTest, 'MiniBatchSize', 1);

YTestArray = YTest{1};
YPredArray = YPred{1};

%% Calculate Error

predictionError = YTestArray - YPredArray;

MSE = mean(predictionError.^2);
RMSE = sqrt(MSE);
MAE = mean(abs(predictionError));

fprintf('\nPrediction Performance:\n');
fprintf('MSE  = %.6f\n', MSE);
fprintf('RMSE = %.6f\n', RMSE);
fprintf('MAE  = %.6f\n', MAE);

%% Plot Prediction Results

figure('Color', 'w', 'Name', 'LSTM Prediction Results');
plot(YTestArray, 'LineWidth', 1.8);
hold on;
plot(YPredArray, '--', 'LineWidth', 1.8);
grid on;
xlabel('Time Step');
ylabel('Normalized Output');
title('LSTM Prediction Result');
legend('Actual Output', 'Predicted Output', 'Location', 'best');

saveas(gcf, fullfile(resultsFolder, 'LSTM_Prediction_Result.png'));
savefig(gcf, fullfile(resultsFolder, 'LSTM_Prediction_Result.fig'));

%% Plot Prediction Error

figure('Color', 'w', 'Name', 'LSTM Prediction Error');
plot(predictionError, 'LineWidth', 1.5);
grid on;
xlabel('Time Step');
ylabel('Prediction Error');
title('LSTM Prediction Error');

saveas(gcf, fullfile(resultsFolder, 'LSTM_Prediction_Error.png'));
savefig(gcf, fullfile(resultsFolder, 'LSTM_Prediction_Error.fig'));

%% Save MATLAB Results

save(fullfile(resultsFolder, 'Processed_LSTM_Data.mat'), ...
    'fileName', ...
    'sheetList', ...
    'mainSheetName', ...
    'allSheetsData', ...
    'sheetSummaryTable', ...
    'numericDataTable', ...
    'dataMatrixOriginal', ...
    'dataMatrix', ...
    'dataNormalized', ...
    'dataMean', ...
    'dataStd', ...
    'inputData', ...
    'outputData', ...
    'XTrain', ...
    'YTrain', ...
    'XTest', ...
    'YTest', ...
    'net', ...
    'YPred', ...
    'MSE', ...
    'RMSE', ...
    'MAE');

fprintf('\nProcessed data and trained LSTM network saved successfully.\n');

%% Create README File

readmeText = [
"# LSTM Excel Data Training MATLAB" newline newline ...
"This repository contains a MATLAB based workflow for reading Excel datasets, analyzing sheet information, preprocessing numerical data, and training an LSTM network for data driven modeling and time series prediction." newline newline ...
"## Main Features" newline newline ...
"- Reads Excel files with multiple sheets" newline ...
"- Displays sheet names, row counts, column counts, and variable names" newline ...
"- Extracts numeric columns automatically" newline ...
"- Checks missing values" newline ...
"- Removes incomplete rows" newline ...
"- Normalizes data for neural network training" newline ...
"- Prepares sequence data for LSTM training" newline ...
"- Trains an LSTM regression network" newline ...
"- Saves prediction plots and processed MATLAB data" newline ...
"- Pushes code, data, and results to GitHub" newline newline ...
"## Excel File" newline newline ...
"The current Excel file used in this project is:" newline newline ...
"`data- 5-26.xlsx`" newline newline ...
"## Main MATLAB File" newline newline ...
"`Main_Read_Excel_Train_LSTM.m`" newline newline ...
"## Notes" newline newline ...
"By default, the code uses the first sheet in the Excel file for LSTM training. The last numeric column is considered the output, and all previous numeric columns are considered input features. This can be changed inside the MATLAB code depending on the structure of the dataset." newline
];

fid = fopen('README.md', 'w');
fprintf(fid, '%s', readmeText);
fclose(fid);

fprintf('\nREADME.md file created successfully.\n');

%% ============================================================
%  Push Data, Code, and Results to GitHub
%% ============================================================

fprintf('\nPreparing GitHub push...\n');

% Check whether this folder is already a Git repository
[gitStatusCode, ~] = system('git status');

if gitStatusCode ~= 0
    fprintf('This folder is not a Git repository. Initializing Git...\n');
    system('git init');
end

% Check existing remote
[~, remoteText] = system('git remote -v');

if contains(remoteText, 'origin')
    system(['git remote set-url origin ', repoURL]);
else
    system(['git remote add origin ', repoURL]);
end

% Add files
system('git add "data- 5-26.xlsx"');
system('git add *.m');
system('git add README.md');
system(['git add ', resultsFolder]);

% Commit changes
commitMessage = 'Add Excel data reading and LSTM training workflow';
[commitStatus, commitOutput] = system(['git commit -m "', commitMessage, '"']);

disp(commitOutput);

% Set branch and push
system('git branch -M main');

[pushStatus, pushOutput] = system('git push -u origin main');

disp(pushOutput);

if pushStatus == 0
    fprintf('\nFiles pushed to GitHub successfully.\n');
else
    fprintf('\nGitHub push did not complete successfully.\n');
    fprintf('Check your GitHub login, internet connection, or repository permission.\n');
end