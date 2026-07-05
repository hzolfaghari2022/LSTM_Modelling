%% ESA COMSOL Single Actuator Data Visualization and LSTM Modeling
% Purpose:
%   Step 1: Read the COMSOL spreadsheet for one ESA actuator.
%   Step 2: Plot each signal one by one with physically meaningful axes.
%   Step 3: Build a first LSTM based dynamic model for a SINGLE actuator.
%   Step 4: Predict displacement and coil force from current history.
%   Step 5: Save all report-ready figures, cleaned data, trained model, and prediction results.
%   Step 6: Optionally push the current project outputs to the correct GitHub repo.
%
% Important:
%   This is NOT the 80 actuator state space network model.
%   This script is for one actuator only, because the COMSOL data contains
%   one actuator current, one force response, and one displacement response.
%
% LSTM modeling idea:
%   The recurrent model learns this dynamic mapping:
%
%       current history  --->  displacement response and force response
%
%   In other words, the LSTM approximates the input output behavior of the
%   single actuator without explicitly writing the mass spring damper state
%   space equations.
%
% File expected in the same folder:
%   ESA-COMSOL_Data_05_22_2026.xlsx
%
% GitHub note:
%   This script uses SSH authentication for GitHub.
%   After you add your SSH public key to GitHub once, MATLAB can push without
%   asking for a personal access token.

clear; clc; close all;

%% ============================================================
% User settings
% =============================================================
excelFile = 'ESA-COMSOL_Data_05_22_2026.xlsx';

figFolder = fullfile(pwd, 'figures');
if ~exist(figFolder, 'dir')
    mkdir(figFolder);
end

% LSTM settings
trainLSTM = true;
trainRatio = 0.75;              % First 75 percent of the time history is used for training.
numHiddenUnits = 64;
maxEpochs = 600;
initialLearnRate = 1e-3;
gradientThreshold = 1.0;

% GitHub push settings
doGitHubPush = true;            % Set to false if you only want to run locally.
repoOwner = 'hzolfaghari2022';
repoName  = 'LSTM_Modelling';
branchName = 'main';
targetFolderName = 'ESA_Single_Actuator_LSTM_Model';

if ~isfile(excelFile)
    error('The Excel file was not found in the current folder: %s', excelFile);
end

%% ============================================================
% Read raw data from each sheet
% =============================================================
% Displacement sheet has metadata in rows 1 to 4. The real header starts in row 5.
D = readtable(excelFile, 'Sheet', 'Displacement', 'Range', 'A5:B565', ...
    'VariableNamingRule', 'preserve');

F = readtable(excelFile, 'Sheet', 'Force', 'Range', 'A2:C519', ...
    'VariableNamingRule', 'preserve');

C = readtable(excelFile, 'Sheet', 'Current', 'Range', 'A2:B519', ...
    'VariableNamingRule', 'preserve');

% Rename columns to clean variable names.
D.Properties.VariableNames = {'Time_s', 'Displacement_mm'};
F.Properties.VariableNames = {'Time_s', 'CoilForce_N', 'WeightLoad_N'};
C.Properties.VariableNames = {'Time_s', 'Current_A'};

%% ============================================================
% Clean duplicate time stamps and sort data
% =============================================================
D = cleanTimeTable(D, 'Displacement_mm');
F = cleanTimeTable(F, {'CoilForce_N', 'WeightLoad_N'});
C = cleanTimeTable(C, 'Current_A');

%% ============================================================
% Create synchronized signals for comparison and LSTM modeling
% =============================================================
% Force and current use the same main time grid. Displacement is interpolated
% onto the force time vector only for combined plots and model training.
t = F.Time_s;
currentOnGrid = interp1(C.Time_s, C.Current_A, t, 'linear', 'extrap');
dispOnGrid = interp1(D.Time_s, D.Displacement_mm, t, 'linear', 'extrap');
coilForce = F.CoilForce_N;
weightLoad = F.WeightLoad_N;
netForce = coilForce - weightLoad;

% Basic time information for reporting.
dt_vec = diff(t);
dt_mean = mean(dt_vec, 'omitnan');
fprintf('\n====================================================\n');
fprintf('ESA single actuator dataset summary\n');
fprintf('====================================================\n');
fprintf('Number of synchronized samples: %d\n', numel(t));
fprintf('Time range: %.6f s to %.6f s\n', min(t), max(t));
fprintf('Mean sampling time: %.6f s\n', dt_mean);
fprintf('Mean sampling time: %.3f ms\n', dt_mean * 1000);
fprintf('Current range: %.6f A to %.6f A\n', min(currentOnGrid), max(currentOnGrid));
fprintf('Displacement range: %.6f mm to %.6f mm\n', min(dispOnGrid), max(dispOnGrid));
fprintf('Coil force range: %.6f N to %.6f N\n', min(coilForce), max(coilForce));
fprintf('====================================================\n');

%% ============================================================
% Figure 1: Current input versus time
% =============================================================
figure('Color','w');
plot(C.Time_s, C.Current_A, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)');
ylabel('Coil current (A)');
title('Input Current Profile for Single ESA Actuator');
localSavePng(gcf, fullfile(figFolder, 'Fig01_Current_Time.png'));

%% Figure 2: Coil force versus time
figure('Color','w');
plot(F.Time_s, F.CoilForce_N, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)');
ylabel('Coil force (N)');
title('Electromagnetic Coil Force Response');
localSavePng(gcf, fullfile(figFolder, 'Fig02_CoilForce_Time.png'));

%% Figure 3: Weight/load force versus time
figure('Color','w');
plot(F.Time_s, F.WeightLoad_N, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)');
ylabel('Weight/load force (N)');
title('Weight/Load Force');
localSavePng(gcf, fullfile(figFolder, 'Fig03_Weight_Time.png'));

%% Figure 4: Net force estimate versus time
figure('Color','w');
plot(t, netForce, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)');
ylabel('Net force estimate (N)');
title('Net Force Estimate: Coil Force Minus Weight/Load');
localSavePng(gcf, fullfile(figFolder, 'Fig04_NetForce_Time.png'));

%% Figure 5: Displacement versus time
figure('Color','w');
plot(D.Time_s, D.Displacement_mm, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)');
ylabel('Z displacement (mm)');
title('Z Displacement Response of Single ESA Actuator');
localSavePng(gcf, fullfile(figFolder, 'Fig05_Displacement_Time.png'));

%% Figure 6: Synchronized signals versus time
figure('Color','w');
plot(t, currentOnGrid, 'LineWidth', 1.5); hold on;
plot(t, coilForce, 'LineWidth', 1.5);
plot(t, dispOnGrid, 'LineWidth', 1.5);
grid on; box on;
xlabel('Time (s)');
ylabel('Signal value with original units');
title('Synchronized Single Actuator Signals');
legend('Current (A)', 'Coil force (N)', 'Displacement (mm)', 'Location', 'best');
localSavePng(gcf, fullfile(figFolder, 'Fig06_SynchronizedSignals_Time.png'));

%% Figure 7: Trigger and early transient zoom
figure('Color','w');
plot(1000*t, currentOnGrid, 'LineWidth', 1.5); hold on;
plot(1000*t, coilForce, 'LineWidth', 1.5);
plot(1000*t, dispOnGrid, 'LineWidth', 1.5);
grid on; box on;
xlim([0 30]);
xlabel('Time (ms)');
ylabel('Signal value with original units');
title('Trigger and Early Transient Region, 0 to 30 ms');
legend('Current (A)', 'Coil force (N)', 'Displacement (mm)', 'Location', 'best');
localSavePng(gcf, fullfile(figFolder, 'Fig07_TriggerZoom_0_30ms.png'));

%% Figure 8: Force versus current
figure('Color','w');
plot(currentOnGrid, coilForce, 'LineWidth', 1.8);
grid on; box on;
xlabel('Coil current (A)');
ylabel('Coil force (N)');
title('Coil Force Versus Input Current');
localSavePng(gcf, fullfile(figFolder, 'Fig08_Force_Current.png'));

%% Figure 9: Displacement versus current
figure('Color','w');
plot(currentOnGrid, dispOnGrid, 'LineWidth', 1.8);
grid on; box on;
xlabel('Coil current (A)');
ylabel('Z displacement (mm)');
title('Displacement Versus Input Current');
localSavePng(gcf, fullfile(figFolder, 'Fig09_Displacement_Current.png'));

%% Figure 10: Force versus displacement
figure('Color','w');
plot(dispOnGrid, coilForce, 'LineWidth', 1.8);
grid on; box on;
xlabel('Z displacement (mm)');
ylabel('Coil force (N)');
title('Force Versus Displacement');
localSavePng(gcf, fullfile(figFolder, 'Fig10_Force_Displacement.png'));


%% ============================================================
% Additional report-ready data summary figure
% =============================================================
% This figure is created specifically for the Overleaf reports. It summarizes
% the main measured/simulated signals in one compact panel.
figure('Color','w', 'Position', [100 100 1000 750]);
subplot(3,1,1);
plot(t, currentOnGrid, 'LineWidth', 1.8);
grid on; box on;
ylabel('Current (A)');
title('Report Summary of Single Actuator Dataset');

subplot(3,1,2);
plot(t, coilForce, 'LineWidth', 1.8); hold on;
plot(t, weightLoad, '--', 'LineWidth', 1.4);
grid on; box on;
ylabel('Force (N)');
legend('Coil force', 'Weight/load', 'Location', 'best');

subplot(3,1,3);
plot(t, dispOnGrid, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)');
ylabel('Displacement (mm)');
localSavePng(gcf, fullfile(figFolder, 'Fig15_Report_Data_Summary.png'));

%% ============================================================
% Save cleaned data
% =============================================================
cleanedData = table(t, currentOnGrid, coilForce, weightLoad, netForce, dispOnGrid, ...
    'VariableNames', {'Time_s','Current_A','CoilForce_N','WeightLoad_N','NetForce_N','Displacement_mm'});

writetable(cleanedData, 'ESA_cleaned_synchronized_data.csv');
save('ESA_cleaned_synchronized_data.mat', 'D', 'F', 'C', 'cleanedData');

disp('Data visualization completed successfully. Figures were saved in the figures folder.');

%% ============================================================
% LSTM dynamic model for ONE actuator
% =============================================================
% This section trains a sequence to sequence LSTM model for one actuator.
%
% Input:
%   X(k) = Current_A(k)
%
% Outputs:
%   Y1(k) = Displacement_mm(k)
%   Y2(k) = CoilForce_N(k)
%
% The LSTM therefore learns a dynamic input output model:
%
%   [Displacement_mm(k), CoilForce_N(k)] = f_LSTM(Current_A(1:k))
%
% This is the single actuator analogue of the network actuator model. The
% difference is that the state space network has many actuator inputs and
% many internal states, while this dataset contains only one actuator signal.

if trainLSTM
    fprintf('\n====================================================\n');
    fprintf('Training single actuator LSTM model\n');
    fprintf('====================================================\n');

    if exist('trainNetwork', 'file') ~= 2
        error(['Deep Learning Toolbox is required for trainNetwork. ', ...
               'Install or enable Deep Learning Toolbox before running the LSTM section.']);
    end

    % Raw sequence format expected by MATLAB sequence networks:
    % input features x time steps
    Xraw = currentOnGrid(:)';                 % 1 x T
    Yraw = [dispOnGrid(:)'; coilForce(:)'];   % 2 x T

    numSamples = numel(t);
    nTrain = max(10, floor(trainRatio * numSamples));
    nTrain = min(nTrain, numSamples - 5);

    idxTrain = 1:nTrain;
    idxTest = (nTrain + 1):numSamples;

    XTrainRaw = Xraw(:, idxTrain);
    YTrainRaw = Yraw(:, idxTrain);

    XTestRaw = Xraw(:, idxTest);
    YTestRaw = Yraw(:, idxTest);

    % Normalize using training data only.
    muX = mean(XTrainRaw, 2);
    sigX = std(XTrainRaw, 0, 2);
    sigX(sigX == 0) = 1;

    muY = mean(YTrainRaw, 2);
    sigY = std(YTrainRaw, 0, 2);
    sigY(sigY == 0) = 1;

    XTrain = normalizeSeq(XTrainRaw, muX, sigX);
    YTrain = normalizeSeq(YTrainRaw, muY, sigY);

    XTest = normalizeSeq(XTestRaw, muX, sigX);
    YTest = normalizeSeq(YTestRaw, muY, sigY);

    XFull = normalizeSeq(Xraw, muX, sigX);

    %% Figure 16: Train/test split and normalized sequences for report
    figure('Color','w', 'Position', [100 100 1000 750]);
    subplot(3,1,1);
    plot(t, XFull(1,:), 'LineWidth', 1.5); hold on;
    xline(t(nTrain), 'k:', 'Train/Test split', 'LineWidth', 1.2);
    grid on; box on;
    ylabel('Normalized current');
    title('Normalized LSTM Input and Outputs');

    YFullNorm = normalizeSeq(Yraw, muY, sigY);
    subplot(3,1,2);
    plot(t, YFullNorm(1,:), 'LineWidth', 1.5); hold on;
    xline(t(nTrain), 'k:', 'Train/Test split', 'LineWidth', 1.2);
    grid on; box on;
    ylabel('Normalized displacement');

    subplot(3,1,3);
    plot(t, YFullNorm(2,:), 'LineWidth', 1.5); hold on;
    xline(t(nTrain), 'k:', 'Train/Test split', 'LineWidth', 1.2);
    grid on; box on;
    xlabel('Time (s)');
    ylabel('Normalized force');
    localSavePng(gcf, fullfile(figFolder, 'Fig16_LSTM_Normalized_Sequences.png'));

    %% Figure 17: LSTM modeling workflow for report
    figure('Color','w', 'Position', [100 100 1000 350]);
    axis off;
    text(0.08, 0.55, {'Input sequence', 'Current history', 'I(1), I(2), ..., I(k)'}, ...
        'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.95 0.95 0.95], 'EdgeColor', 'k', 'Margin', 12);
    text(0.50, 0.55, {'LSTM dynamic memory', 'Learns time-dependent', 'actuator behavior'}, ...
        'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.95 0.95 0.95], 'EdgeColor', 'k', 'Margin', 12);
    text(0.88, 0.55, {'Output sequence', 'Displacement z(k)', 'Coil force F(k)'}, ...
        'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.95 0.95 0.95], 'EdgeColor', 'k', 'Margin', 12);
    annotation('arrow', [0.22 0.39], [0.55 0.55], 'LineWidth', 1.8);
    annotation('arrow', [0.62 0.77], [0.55 0.55], 'LineWidth', 1.8);
    text(0.50, 0.18, sprintf('Architecture: sequence input layer + LSTM (%d hidden units) + fully connected layers + regression output', numHiddenUnits), ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    localSavePng(gcf, fullfile(figFolder, 'Fig17_LSTM_Model_Workflow.png'));

    layers = [
        sequenceInputLayer(1, 'Name', 'current_input')
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'lstm_dynamic_memory')
        fullyConnectedLayer(32, 'Name', 'dense_dynamic_features')
        reluLayer('Name', 'relu')
        fullyConnectedLayer(2, 'Name', 'predicted_displacement_and_force')
        regressionLayer('Name', 'regression_output')];

    options = trainingOptions('adam', ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', 1, ...
        'InitialLearnRate', initialLearnRate, ...
        'GradientThreshold', gradientThreshold, ...
        'Shuffle', 'never', ...
        'ValidationData', {{XTest}, {YTest}}, ...
        'ValidationFrequency', 25, ...
        'Verbose', true, ...
        'Plots', 'training-progress');

    % Train the LSTM. Cell arrays are used because the data are time sequences.
    [netLSTM, trainInfo] = trainNetwork({XTrain}, {YTrain}, layers, options);

    % Predict the entire time history.
    YPredFullNormCell = predict(netLSTM, {XFull}, 'MiniBatchSize', 1);
    if iscell(YPredFullNormCell)
        YPredFullNorm = YPredFullNormCell{1};
    else
        YPredFullNorm = YPredFullNormCell;
    end

    YPredFull = denormalizeSeq(YPredFullNorm, muY, sigY);

    predDisplacement_mm = YPredFull(1, :)';
    predCoilForce_N = YPredFull(2, :)';

    trueDisplacement_mm = dispOnGrid(:);
    trueCoilForce_N = coilForce(:);

    displacementError_mm = trueDisplacement_mm - predDisplacement_mm;
    coilForceError_N = trueCoilForce_N - predCoilForce_N;

    trainMask = false(numSamples, 1);
    trainMask(idxTrain) = true;

    testMask = false(numSamples, 1);
    testMask(idxTest) = true;

    lstmMetrics = struct();
    lstmMetrics.RMSE_Displacement_Train_mm = sqrt(mean(displacementError_mm(trainMask).^2));
    lstmMetrics.RMSE_Displacement_Test_mm  = sqrt(mean(displacementError_mm(testMask).^2));
    lstmMetrics.RMSE_Force_Train_N = sqrt(mean(coilForceError_N(trainMask).^2));
    lstmMetrics.RMSE_Force_Test_N  = sqrt(mean(coilForceError_N(testMask).^2));

    lstmMetrics.MAE_Displacement_Train_mm = mean(abs(displacementError_mm(trainMask)));
    lstmMetrics.MAE_Displacement_Test_mm  = mean(abs(displacementError_mm(testMask)));
    lstmMetrics.MAE_Force_Train_N = mean(abs(coilForceError_N(trainMask)));
    lstmMetrics.MAE_Force_Test_N  = mean(abs(coilForceError_N(testMask)));

    fprintf('\nLSTM metrics for single actuator model:\n');
    fprintf('Displacement RMSE train: %.6g mm\n', lstmMetrics.RMSE_Displacement_Train_mm);
    fprintf('Displacement RMSE test : %.6g mm\n', lstmMetrics.RMSE_Displacement_Test_mm);
    fprintf('Coil force RMSE train  : %.6g N\n', lstmMetrics.RMSE_Force_Train_N);
    fprintf('Coil force RMSE test   : %.6g N\n', lstmMetrics.RMSE_Force_Test_N);

    %% Save and plot LSTM training history for report
    trainingHistory = trainingInfoToTable(trainInfo);
    writetable(trainingHistory, 'ESA_single_actuator_LSTM_training_history.csv');

    %% Figure 20: LSTM training RMSE history
    figure('Color','w'); hold on;
    if ismember('TrainingRMSE', trainingHistory.Properties.VariableNames)
        validTrain = isfinite(trainingHistory.TrainingRMSE);
        plot(trainingHistory.Iteration(validTrain), trainingHistory.TrainingRMSE(validTrain), 'LineWidth', 1.8);
    end
    if ismember('ValidationRMSE', trainingHistory.Properties.VariableNames)
        validVal = isfinite(trainingHistory.ValidationRMSE);
        plot(trainingHistory.Iteration(validVal), trainingHistory.ValidationRMSE(validVal), '--', 'LineWidth', 1.8);
    end
    grid on; box on;
    xlabel('Iteration');
    ylabel('RMSE');
    title('LSTM Training and Validation RMSE History');
    legend('Training RMSE', 'Validation RMSE', 'Location', 'best');
    localSavePng(gcf, fullfile(figFolder, 'Fig20_LSTM_Training_RMSE_History.png'));

    %% Figure 21: LSTM training loss history
    figure('Color','w'); hold on;
    if ismember('TrainingLoss', trainingHistory.Properties.VariableNames)
        validTrain = isfinite(trainingHistory.TrainingLoss);
        plot(trainingHistory.Iteration(validTrain), trainingHistory.TrainingLoss(validTrain), 'LineWidth', 1.8);
    end
    if ismember('ValidationLoss', trainingHistory.Properties.VariableNames)
        validVal = isfinite(trainingHistory.ValidationLoss);
        plot(trainingHistory.Iteration(validVal), trainingHistory.ValidationLoss(validVal), '--', 'LineWidth', 1.8);
    end
    grid on; box on;
    xlabel('Iteration');
    ylabel('Loss');
    title('LSTM Training and Validation Loss History');
    legend('Training loss', 'Validation loss', 'Location', 'best');
    localSavePng(gcf, fullfile(figFolder, 'Fig21_LSTM_Training_Loss_History.png'));

    %% Save LaTeX table of LSTM performance metrics for Overleaf
    writeLSTMMetricsTable('ESA_single_actuator_LSTM_metrics_table.tex', lstmMetrics);
    fprintf('LSTM training history and Overleaf metrics table saved successfully.\n');

    %% Figure 11: LSTM displacement prediction
    figure('Color','w');
    plot(t, trueDisplacement_mm, 'LineWidth', 1.8); hold on;
    plot(t, predDisplacement_mm, '--', 'LineWidth', 1.8);
    xline(t(nTrain), 'k:', 'Train/Test split', 'LineWidth', 1.2);
    grid on; box on;
    xlabel('Time (s)');
    ylabel('Z displacement (mm)');
    title('Single Actuator LSTM Model: Displacement Prediction');
    legend('COMSOL displacement', 'LSTM prediction', 'Train/Test split', 'Location', 'best');
    localSavePng(gcf, fullfile(figFolder, 'Fig11_LSTM_Displacement_Prediction.png'));

    %% Figure 12: LSTM force prediction
    figure('Color','w');
    plot(t, trueCoilForce_N, 'LineWidth', 1.8); hold on;
    plot(t, predCoilForce_N, '--', 'LineWidth', 1.8);
    xline(t(nTrain), 'k:', 'Train/Test split', 'LineWidth', 1.2);
    grid on; box on;
    xlabel('Time (s)');
    ylabel('Coil force (N)');
    title('Single Actuator LSTM Model: Coil Force Prediction');
    legend('COMSOL coil force', 'LSTM prediction', 'Train/Test split', 'Location', 'best');
    localSavePng(gcf, fullfile(figFolder, 'Fig12_LSTM_Force_Prediction.png'));

    %% Figure 13: LSTM prediction errors
    figure('Color','w');
    subplot(2,1,1);
    plot(t, displacementError_mm, 'LineWidth', 1.6); hold on;
    xline(t(nTrain), 'k:', 'Train/Test split', 'LineWidth', 1.2);
    grid on; box on;
    xlabel('Time (s)');
    ylabel('Error (mm)');
    title('Displacement Prediction Error: COMSOL minus LSTM');

    subplot(2,1,2);
    plot(t, coilForceError_N, 'LineWidth', 1.6); hold on;
    xline(t(nTrain), 'k:', 'Train/Test split', 'LineWidth', 1.2);
    grid on; box on;
    xlabel('Time (s)');
    ylabel('Error (N)');
    title('Coil Force Prediction Error: COMSOL minus LSTM');
    localSavePng(gcf, fullfile(figFolder, 'Fig13_LSTM_Prediction_Errors.png'));

    %% Figure 14: Trigger zoom with LSTM predictions
    zoomMask = t <= 0.03;
    figure('Color','w');

    yyaxis left;
    plot(1000*t(zoomMask), trueDisplacement_mm(zoomMask), 'LineWidth', 1.7); hold on;
    plot(1000*t(zoomMask), predDisplacement_mm(zoomMask), '--', 'LineWidth', 1.7);
    ylabel('Displacement (mm)');

    yyaxis right;
    plot(1000*t(zoomMask), currentOnGrid(zoomMask), ':', 'LineWidth', 1.7);
    ylabel('Current (A)');

    grid on; box on;
    xlabel('Time (ms)');
    title('Early Transient: Current Trigger and LSTM Displacement Prediction');
    legend('COMSOL displacement', 'LSTM displacement', 'Current input', 'Location', 'best');
    localSavePng(gcf, fullfile(figFolder, 'Fig14_LSTM_TriggerZoom.png'));

    %% Figure 18: Error metrics for report
    figure('Color','w', 'Position', [100 100 1000 420]);
    subplot(1,2,1);
    bar([lstmMetrics.RMSE_Displacement_Train_mm, lstmMetrics.RMSE_Displacement_Test_mm; ...
         lstmMetrics.MAE_Displacement_Train_mm,  lstmMetrics.MAE_Displacement_Test_mm]);
    grid on; box on;
    set(gca, 'XTickLabel', {'RMSE', 'MAE'});
    ylabel('Displacement error (mm)');
    title('Displacement Prediction Metrics');
    legend('Train', 'Test', 'Location', 'best');

    subplot(1,2,2);
    bar([lstmMetrics.RMSE_Force_Train_N, lstmMetrics.RMSE_Force_Test_N; ...
         lstmMetrics.MAE_Force_Train_N,  lstmMetrics.MAE_Force_Test_N]);
    grid on; box on;
    set(gca, 'XTickLabel', {'RMSE', 'MAE'});
    ylabel('Force error (N)');
    title('Force Prediction Metrics');
    legend('Train', 'Test', 'Location', 'best');
    localSavePng(gcf, fullfile(figFolder, 'Fig18_LSTM_Error_Metrics.png'));

    %% Figure 19: Parity plots for report
    figure('Color','w', 'Position', [100 100 1000 420]);
    subplot(1,2,1);
    plot(trueDisplacement_mm, predDisplacement_mm, '.', 'MarkerSize', 10); hold on;
    minD = min([trueDisplacement_mm; predDisplacement_mm]);
    maxD = max([trueDisplacement_mm; predDisplacement_mm]);
    plot([minD maxD], [minD maxD], 'k--', 'LineWidth', 1.4);
    grid on; box on; axis equal;
    xlabel('COMSOL displacement (mm)');
    ylabel('LSTM displacement (mm)');
    title('Displacement Parity Plot');

    subplot(1,2,2);
    plot(trueCoilForce_N, predCoilForce_N, '.', 'MarkerSize', 10); hold on;
    minF = min([trueCoilForce_N; predCoilForce_N]);
    maxF = max([trueCoilForce_N; predCoilForce_N]);
    plot([minF maxF], [minF maxF], 'k--', 'LineWidth', 1.4);
    grid on; box on; axis equal;
    xlabel('COMSOL coil force (N)');
    ylabel('LSTM coil force (N)');
    title('Coil Force Parity Plot');
    localSavePng(gcf, fullfile(figFolder, 'Fig19_LSTM_Parity_Plots.png'));

    %% Save LSTM results
    lstmResults = table(t, currentOnGrid, ...
        trueDisplacement_mm, predDisplacement_mm, displacementError_mm, ...
        trueCoilForce_N, predCoilForce_N, coilForceError_N, ...
        trainMask, testMask, ...
        'VariableNames', {'Time_s', 'Current_A', ...
        'TrueDisplacement_mm', 'PredictedDisplacement_mm', 'DisplacementError_mm', ...
        'TrueCoilForce_N', 'PredictedCoilForce_N', 'CoilForceError_N', ...
        'IsTrainingSample', 'IsTestingSample'});

    writetable(lstmResults, 'ESA_single_actuator_LSTM_results.csv');

    normalization = struct();
    normalization.muX = muX;
    normalization.sigX = sigX;
    normalization.muY = muY;
    normalization.sigY = sigY;
    normalization.inputName = 'Current_A';
    normalization.outputNames = {'Displacement_mm', 'CoilForce_N'};

    save('ESA_single_actuator_LSTM_model.mat', ...
        'netLSTM', 'layers', 'options', 'trainInfo', 'trainingHistory', 'normalization', ...
        'lstmMetrics', 'lstmResults', 'cleanedData');

    writeLSTMSummary('ESA_single_actuator_LSTM_summary.txt', lstmMetrics, trainRatio, numHiddenUnits, maxEpochs);

    fprintf('Single actuator LSTM model training completed successfully.\n');
    fprintf('LSTM model saved as ESA_single_actuator_LSTM_model.mat\n');
    fprintf('LSTM prediction results saved as ESA_single_actuator_LSTM_results.csv\n');
else
    fprintf('\nLSTM training is disabled. Set trainLSTM = true to train the model.\n');
end


%% ============================================================
% Save report figure inventory for Overleaf
% =============================================================
writeFigureInventory(figFolder, 'ESA_LSTM_Report_Figure_List.txt', 'ESA_LSTM_Report_Figure_Includes.tex');
fprintf('Report figure inventory saved in the figures folder.\n');

%% ============================================================
% GitHub Push Block for LSTM_Modelling Repository
% =============================================================
% This block uses SSH, not a GitHub token.
%
% One time setup required before using this block:
%   1. Create an SSH key.
%   2. Add the public key to GitHub.
%   3. Test in Git Bash:
%        ssh -T git@github.com
%
% If the test says that you successfully authenticated, MATLAB can push
% without asking for a token.

if doGitHubPush
    disp('====================================================');
    disp('Preparing GitHub SSH push for LSTM_Modelling repository');
    disp('====================================================');

    repoURL_SSH = ['git@github.com:' repoOwner '/' repoName '.git'];

    % Folder where the current MATLAB script saved results.
    sourceFolder = pwd;

    % Temporary clean GitHub working folder.
    githubRoot = fullfile(tempdir, 'MATLAB_GitHub_Repos');
    repoFolder = fullfile(githubRoot, repoName);
    targetFolder = fullfile(repoFolder, targetFolderName);

    % Remember current folder and clean up MATLAB path even if an error occurs.
    originalFolder = pwd;
    cleanupObj = onCleanup(@()safeReturnAndDisconnect(originalFolder, repoFolder)); %#ok<NASGU>

    %% Step 1: Remove any previous repo folders from MATLAB path
    fprintf('Removing previous GitHub repo folders from MATLAB path...\n');
    if exist(githubRoot, 'dir')
        oldPaths = genpath(githubRoot);
        if ~isempty(oldPaths)
            rmpath(oldPaths);
        end
    end

    %% Step 2: Create GitHub root folder
    if ~exist(githubRoot, 'dir')
        mkdir(githubRoot);
    end

    %% Step 3: Clone with SSH if local copy does not exist
    if ~exist(repoFolder, 'dir')
        fprintf('Local repository does not exist. Cloning with SSH...\n');

        cloneCommand = sprintf('git clone "%s" "%s"', repoURL_SSH, repoFolder);
        [cloneStatus, cloneOut] = system(cloneCommand);

        if cloneStatus ~= 0
            error(['Git clone failed using SSH.\n\n' ...
                   'Most likely the SSH key is not connected to GitHub yet.\n\n' ...
                   'Fix in Git Bash:\n' ...
                   '  ssh -T git@github.com\n\n' ...
                   'If GitHub does not authenticate you, add ~/.ssh/id_ed25519.pub ' ...
                   'to GitHub > Settings > SSH and GPG keys.\n\n' ...
                   'Git message:\n%s'], cloneOut);
        end
    else
        fprintf('Local repository already exists.\n');
    end

    %% Step 4: Enter repository and verify it is valid
    cd(repoFolder);

    [insideStatus, insideOut] = system('git rev-parse --is-inside-work-tree');
    if insideStatus ~= 0 || ~contains(strtrim(insideOut), 'true')
        error('The local folder exists but is not a valid Git repository: %s', repoFolder);
    end

    %% Step 5: Make sure the remote is the correct SSH repository
    [remoteStatus, remoteOut] = system('git remote get-url origin');

    if remoteStatus ~= 0
        fprintf('No origin remote found. Adding SSH origin...\n');
        system(sprintf('git remote add origin "%s"', repoURL_SSH));
    else
        fprintf('Current remote:\n%s\n', strtrim(remoteOut));

        expectedPart1 = [repoOwner '/' repoName '.git'];
        expectedPart2 = [repoOwner '/' repoName];

        if ~contains(remoteOut, expectedPart1) && ~contains(remoteOut, expectedPart2)
            error(['This local folder is connected to a different GitHub repository.\n' ...
                   'Expected: %s\nActual: %s\nStopping to avoid pushing to the wrong repo.'], ...
                   repoURL_SSH, strtrim(remoteOut));
        end

        system(sprintf('git remote set-url origin "%s"', repoURL_SSH));
    end

    %% Step 6: Pull latest version first
    fprintf('Pulling latest changes from GitHub...\n');

    [branchStatus, branchOut] = system(['git branch -M ' branchName]);
    if branchStatus ~= 0
        error('Could not set branch name to %s:\n%s', branchName, branchOut);
    end

    [pullStatus, pullOut] = system(['git pull origin ' branchName ' --rebase']);
    if pullStatus ~= 0
        fprintf('Warning: git pull had an issue:\n%s\n', pullOut);
        fprintf('Continuing. This can happen if the repository is empty or newly created.\n');
    end

    %% Step 7: Add only this repo temporarily to MATLAB path
    addpath(genpath(repoFolder));

    %% Step 8: Copy current project files into the repo
    fprintf('Copying current project files into the repository...\n');

    if ~exist(targetFolder, 'dir')
        mkdir(targetFolder);
    end

    copyFilesByPattern(sourceFolder, '*.m', targetFolder);
    copyFilesByPattern(sourceFolder, '*.xlsx', targetFolder);
    copyFilesByPattern(sourceFolder, '*.mat', targetFolder);
    copyFilesByPattern(sourceFolder, '*.csv', targetFolder);
    copyFilesByPattern(sourceFolder, '*.pdf', targetFolder);
    copyFilesByPattern(sourceFolder, '*.tex', targetFolder);
    copyFilesByPattern(sourceFolder, '*.txt', targetFolder);

    sourceFigFolder = fullfile(sourceFolder, 'figures');
    targetFigFolder = fullfile(targetFolder, 'figures');

    if exist(sourceFigFolder, 'dir')
        if exist(targetFigFolder, 'dir')
            rmdir(targetFigFolder, 's');
        end
        copyfile(sourceFigFolder, targetFigFolder);
    end

    %% Step 9: Create or update .gitignore
    gitignoreFile = fullfile(repoFolder, '.gitignore');
    appendGitIgnore(gitignoreFile, { ...
        '# MATLAB Drive system files', ...
        '.MATLABDriveTag', ...
        '**/.MATLABDriveTag', ...
        '', ...
        '# MATLAB temporary and autosave files', ...
        '*.asv', ...
        '*.m~', ...
        '', ...
        '# Operating system files', ...
        '.DS_Store', ...
        'Thumbs.db'});

    %% Step 10: Configure Git identity
    system('git config user.name "Hussein Zolfaghari"');
    system('git config user.email "h.zolfaghari2015@gmail.com"');

    %% Step 11: Add, commit, and push using SSH
    fprintf('Committing and pushing files to GitHub using SSH...\n');

    [addStatus, addOut] = system('git add .');
    if addStatus ~= 0
        error('Git add failed:\n%s', addOut);
    end

    system('git status');

    [diffStatus, ~] = system('git diff --cached --quiet');

    if diffStatus == 0
        fprintf('No new changes to commit. Repository is already up to date.\n');
    else
        commitMessage = ['Update single actuator ESA LSTM model - ' ...
                         datestr(now, 'yyyy-mm-dd HH:MM:SS')];

        commitCommand = sprintf('git commit -m "%s"', commitMessage);
        [commitStatus, commitOut] = system(commitCommand);

        if commitStatus ~= 0
            error('Git commit failed:\n%s', commitOut);
        else
            fprintf('Commit completed successfully:\n%s\n', commitMessage);
        end
    end

    pushCommand = ['git push -u origin ' branchName];
    [pushStatus, pushOut] = system(pushCommand);

    if pushStatus ~= 0
        error(['Git push failed using SSH.\n\n' ...
               'If the message says Permission denied publickey, your SSH key is not connected to GitHub yet.\n\n' ...
               'Run this in Git Bash to test:\n' ...
               '  ssh -T git@github.com\n\n' ...
               'Git message:\n%s'], pushOut);
    else
        fprintf('Files pushed successfully to GitHub %s branch.\n', branchName);
    end

    %% Step 12: Disconnect repo path and return
    fprintf('Removing this repository from MATLAB path after push...\n');
    rmpath(genpath(repoFolder));
    cd(sourceFolder);

    disp('====================================================');
    disp('GitHub SSH push completed for LSTM_Modelling repository.');
    disp('MATLAB is disconnected from this repository path.');
    disp('====================================================');
else
    fprintf('\nGitHub push is disabled. Set doGitHubPush = true to push results.\n');
end


%% ============================================================
% Local functions
% =============================================================
function Tclean = cleanTimeTable(T, signalNames)
    T = rmmissing(T);
    T = sortrows(T, 'Time_s');

    if ischar(signalNames) || isstring(signalNames)
        signalNames = cellstr(signalNames);
    end

    [uniqueTime, ~, idx] = unique(T.Time_s);
    Tclean = table(uniqueTime, 'VariableNames', {'Time_s'});

    for k = 1:numel(signalNames)
        name = signalNames{k};
        Tclean.(name) = accumarray(idx, T.(name), [], @mean);
    end
end

function Xn = normalizeSeq(X, mu, sig)
    Xn = (X - mu) ./ sig;
end

function X = denormalizeSeq(Xn, mu, sig)
    X = Xn .* sig + mu;
end

function localSavePng(figHandle, fileName)
    % Robust Overleaf-compatible figure saving.
    % For each requested PNG, this function also saves a PDF and a MATLAB FIG
    % file with the same base name. The PNG is used directly in Overleaf, the
    % PDF is useful for vector-style report export, and the FIG file allows
    % later editing in MATLAB.
    [folderPath, baseName, ~] = fileparts(fileName);
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end

    pngName = fullfile(folderPath, [baseName, '.png']);
    pdfName = fullfile(folderPath, [baseName, '.pdf']);
    figName = fullfile(folderPath, [baseName, '.fig']);

    try
        set(figHandle, 'Color', 'w');
        set(findall(figHandle, '-property', 'FontName'), 'FontName', 'Times New Roman');
        set(findall(figHandle, '-property', 'FontSize'), 'FontSize', 12);
        drawnow;

        exportgraphics(figHandle, pngName, 'Resolution', 300);
    catch
        try
            print(figHandle, pngName, '-dpng', '-r300');
        catch
            fr = getframe(figHandle);
            [img, ~] = frame2im(fr);
            imwrite(img, pngName, 'png');
        end
    end

    try
        exportgraphics(figHandle, pdfName, 'ContentType', 'vector');
    catch
        try
            print(figHandle, pdfName, '-dpdf', '-bestfit');
        catch
            warning('Could not save PDF version of figure: %s', pdfName);
        end
    end

    try
        savefig(figHandle, figName);
    catch
        warning('Could not save MATLAB FIG version of figure: %s', figName);
    end

    try
        imfinfo(pngName);
    catch
        warning('The saved PNG was not readable. Rewriting using imwrite: %s', pngName);
        fr = getframe(figHandle);
        [img, ~] = frame2im(fr);
        imwrite(img, pngName, 'png');
    end
end

function writeLSTMSummary(fileName, metrics, trainRatio, numHiddenUnits, maxEpochs)
    fid = fopen(fileName, 'w');
    if fid < 0
        warning('Could not write LSTM summary file.');
        return;
    end

    fprintf(fid, 'Single Actuator ESA LSTM Model Summary\n');
    fprintf(fid, '=====================================\n\n');
    fprintf(fid, 'Model purpose:\n');
    fprintf(fid, 'The LSTM learns a dynamic input output model for one actuator.\n');
    fprintf(fid, 'Input: current history.\n');
    fprintf(fid, 'Outputs: displacement and coil force.\n\n');

    fprintf(fid, 'Training settings:\n');
    fprintf(fid, 'Training ratio: %.2f\n', trainRatio);
    fprintf(fid, 'Hidden units: %d\n', numHiddenUnits);
    fprintf(fid, 'Max epochs: %d\n\n', maxEpochs);

    fprintf(fid, 'Performance metrics:\n');
    fprintf(fid, 'Displacement RMSE train: %.8g mm\n', metrics.RMSE_Displacement_Train_mm);
    fprintf(fid, 'Displacement RMSE test : %.8g mm\n', metrics.RMSE_Displacement_Test_mm);
    fprintf(fid, 'Force RMSE train       : %.8g N\n', metrics.RMSE_Force_Train_N);
    fprintf(fid, 'Force RMSE test        : %.8g N\n', metrics.RMSE_Force_Test_N);
    fprintf(fid, 'Displacement MAE train : %.8g mm\n', metrics.MAE_Displacement_Train_mm);
    fprintf(fid, 'Displacement MAE test  : %.8g mm\n', metrics.MAE_Displacement_Test_mm);
    fprintf(fid, 'Force MAE train        : %.8g N\n', metrics.MAE_Force_Train_N);
    fprintf(fid, 'Force MAE test         : %.8g N\n', metrics.MAE_Force_Test_N);

    fclose(fid);
end

function copyFilesByPattern(sourceFolder, pattern, targetFolder)
    files = dir(fullfile(sourceFolder, pattern));
    for k = 1:numel(files)
        if ~files(k).isdir
            copyfile(fullfile(sourceFolder, files(k).name), targetFolder);
        end
    end
end

function appendGitIgnore(gitignoreFile, linesToAdd)
    existingText = "";
    if isfile(gitignoreFile)
        existingText = string(fileread(gitignoreFile));
    end

    fid = fopen(gitignoreFile, 'a');
    if fid < 0
        error('Could not open .gitignore for writing.');
    end

    for k = 1:numel(linesToAdd)
        thisLine = string(linesToAdd{k});
        if strlength(thisLine) == 0
            fprintf(fid, '\n');
        elseif ~contains(existingText, thisLine)
            fprintf(fid, '%s\n', thisLine);
        end
    end

    fclose(fid);
end



function trainingHistory = trainingInfoToTable(trainInfo)
    % Convert MATLAB trainNetwork info output to a clean table for reports.
    preferredFields = {'Iteration', 'Epoch', 'TrainingLoss', 'TrainingRMSE', ...
        'ValidationLoss', 'ValidationRMSE', 'BaseLearnRate'};

    availableFields = fieldnames(trainInfo);
    usableFields = {};
    maxLen = 0;

    for k = 1:numel(preferredFields)
        name = preferredFields{k};
        if ismember(name, availableFields)
            value = trainInfo.(name);
            if isnumeric(value) || islogical(value)
                value = value(:);
                if numel(value) > 1
                    usableFields{end+1} = name; %#ok<AGROW>
                    maxLen = max(maxLen, numel(value));
                end
            end
        end
    end

    if maxLen == 0
        trainingHistory = table((1:0)', 'VariableNames', {'Iteration'});
        return;
    end

    if ismember('Iteration', usableFields)
        iteration = trainInfo.Iteration(:);
        if numel(iteration) ~= maxLen
            iteration = (1:maxLen)';
        end
    else
        iteration = (1:maxLen)';
    end

    trainingHistory = table(iteration, 'VariableNames', {'Iteration'});

    for k = 1:numel(usableFields)
        name = usableFields{k};
        if strcmp(name, 'Iteration')
            continue;
        end
        value = trainInfo.(name);
        value = value(:);
        if numel(value) < maxLen
            value(end+1:maxLen, 1) = NaN; %#ok<AGROW>
        elseif numel(value) > maxLen
            value = value(1:maxLen);
        end
        trainingHistory.(name) = value;
    end
end

function writeLSTMMetricsTable(fileName, metrics)
    fid = fopen(fileName, 'w');
    if fid < 0
        warning('Could not write LSTM metrics LaTeX table.');
        return;
    end

    fprintf(fid, '\\begin{table}[H]\n');
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{Prediction performance of the single actuator LSTM model.}\n');
    fprintf(fid, '\\label{tab:lstm_prediction_metrics}\n');
    fprintf(fid, '\\begin{tabular}{l c}\n');
    fprintf(fid, '\\hline\n');
    fprintf(fid, 'Metric & Value \\\\ \n');
    fprintf(fid, '\\hline\n');
    fprintf(fid, 'Displacement RMSE, training & %.6g mm \\\\ \n', metrics.RMSE_Displacement_Train_mm);
    fprintf(fid, 'Displacement RMSE, testing & %.6g mm \\\\ \n', metrics.RMSE_Displacement_Test_mm);
    fprintf(fid, 'Coil force RMSE, training & %.6g N \\\\ \n', metrics.RMSE_Force_Train_N);
    fprintf(fid, 'Coil force RMSE, testing & %.6g N \\\\ \n', metrics.RMSE_Force_Test_N);
    fprintf(fid, 'Displacement MAE, training & %.6g mm \\\\ \n', metrics.MAE_Displacement_Train_mm);
    fprintf(fid, 'Displacement MAE, testing & %.6g mm \\\\ \n', metrics.MAE_Displacement_Test_mm);
    fprintf(fid, 'Coil force MAE, training & %.6g N \\\\ \n', metrics.MAE_Force_Train_N);
    fprintf(fid, 'Coil force MAE, testing & %.6g N \\\\ \n', metrics.MAE_Force_Test_N);
    fprintf(fid, '\\hline\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table}\n');

    fclose(fid);
end

function writeFigureInventory(figFolder, listFileName, texFileName)
    % Create a text list and a LaTeX include file for all PNG figures.
    pngFiles = dir(fullfile(figFolder, 'Fig*.png'));
    [~, order] = sort({pngFiles.name});
    pngFiles = pngFiles(order);

    listPath = fullfile(figFolder, listFileName);
    texPath = fullfile(figFolder, texFileName);

    fid = fopen(listPath, 'w');
    if fid >= 0
        fprintf(fid, 'Report-ready figures saved for Overleaf\n');
        fprintf(fid, '=======================================\n\n');
        for k = 1:numel(pngFiles)
            fprintf(fid, '%02d. %s\n', k, pngFiles(k).name);
        end
        fclose(fid);
    end

    fid = fopen(texPath, 'w');
    if fid >= 0
        fprintf(fid, '%% Auto-generated LaTeX figure include file.\n');
        fprintf(fid, '%% Copy selected blocks into your Overleaf report.\n\n');
        for k = 1:numel(pngFiles)
            [~, baseName, ~] = fileparts(pngFiles(k).name);
            captionText = strrep(baseName, '_', ' ');
            fprintf(fid, '\\begin{figure}[H]\n');
            fprintf(fid, '    \\centering\n');
            fprintf(fid, '    \\includegraphics[width=0.92\\textwidth]{figures/%s}\n', pngFiles(k).name);
            fprintf(fid, '    \\caption{%s.}\n', captionText);
            fprintf(fid, '    \\label{fig:%s}\n', lower(baseName));
            fprintf(fid, '\\end{figure}\n\n');
        end
        fclose(fid);
    end
end

function safeReturnAndDisconnect(originalFolder, repoFolder)
    try
        if exist(repoFolder, 'dir')
            rmpath(genpath(repoFolder));
        end
    catch
    end

    try
        if exist(originalFolder, 'dir')
            cd(originalFolder);
        end
    catch
    end
end
