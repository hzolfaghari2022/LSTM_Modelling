%% ESA COMSOL Data Visualization Only
% Purpose:
%   Step 1: Read the COMSOL spreadsheet.
%   Step 2: Plot each signal one by one with physically meaningful axes.
%   Step 3: Save figures for the Overleaf report.
%   Step 4: Push the current project outputs to the correct GitHub repo.
%
% Important:
%   This script does NOT train the LSTM yet. It only organizes and visualizes
%   the data so the supervisor can understand what the dataset contains.
%
% File expected in the same folder:
%   ESA-COMSOL_Data_05_22_2026.xlsx
%
% GitHub note:
%   If GitHub returns error 403, the code is usually correct but the token
%   does not have write permission to the target repository. Create a new
%   token with Contents: Read and write access for this repository.

clear; clc; close all;

%% User settings
excelFile = 'ESA-COMSOL_Data_05_22_2026.xlsx';
figFolder = fullfile(pwd, 'figures');

if ~isfile(excelFile)
    error('The Excel file was not found in the current folder: %s', excelFile);
end

if ~exist(figFolder, 'dir')
    mkdir(figFolder);
end

%% Read raw data from each sheet
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

%% Clean duplicate time stamps and sort data
D = cleanTimeTable(D, 'Displacement_mm');
F = cleanTimeTable(F, {'CoilForce_N', 'WeightLoad_N'});
C = cleanTimeTable(C, 'Current_A');

%% Create synchronized signals for comparison plots
% Force and current use the same main time grid. Displacement is interpolated
% onto the force time vector only for combined plots.
t = F.Time_s;
currentOnGrid = interp1(C.Time_s, C.Current_A, t, 'linear', 'extrap');
dispOnGrid = interp1(D.Time_s, D.Displacement_mm, t, 'linear', 'extrap');
coilForce = F.CoilForce_N;
weightLoad = F.WeightLoad_N;
netForce = coilForce - weightLoad;

%% Figure 1: Current input versus time
figure('Color','w');
plot(C.Time_s, C.Current_A, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)'); ylabel('Coil current (A)');
title('Input current profile');
exportgraphics(gcf, fullfile(figFolder, 'Fig01_Current_Time.png'), 'Resolution', 300);

%% Figure 2: Coil force versus time
figure('Color','w');
plot(F.Time_s, F.CoilForce_N, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)'); ylabel('Coil force (N)');
title('Electromagnetic coil force response');
exportgraphics(gcf, fullfile(figFolder, 'Fig02_CoilForce_Time.png'), 'Resolution', 300);

%% Figure 3: Weight/load force versus time
figure('Color','w');
plot(F.Time_s, F.WeightLoad_N, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)'); ylabel('Weight/load force (N)');
title('Constant weight of coil and load');
exportgraphics(gcf, fullfile(figFolder, 'Fig03_Weight_Time.png'), 'Resolution', 300);

%% Figure 4: Net force estimate versus time
figure('Color','w');
plot(t, netForce, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)'); ylabel('Net force estimate (N)');
title('Net force estimate: coil force minus weight/load');
exportgraphics(gcf, fullfile(figFolder, 'Fig04_NetForce_Time.png'), 'Resolution', 300);

%% Figure 5: Displacement versus time
figure('Color','w');
plot(D.Time_s, D.Displacement_mm, 'LineWidth', 1.8);
grid on; box on;
xlabel('Time (s)'); ylabel('Z displacement (mm)');
title('Z displacement response');
exportgraphics(gcf, fullfile(figFolder, 'Fig05_Displacement_Time.png'), 'Resolution', 300);

%% Figure 6: Synchronized signals versus time
figure('Color','w');
plot(t, currentOnGrid, 'LineWidth', 1.5); hold on;
plot(t, coilForce, 'LineWidth', 1.5);
plot(t, dispOnGrid, 'LineWidth', 1.5);
grid on; box on;
xlabel('Time (s)'); ylabel('Signal value with original units');
title('Synchronized signals on the force time grid');
legend('Current (A)', 'Coil force (N)', 'Displacement (mm)', 'Location', 'best');
exportgraphics(gcf, fullfile(figFolder, 'Fig06_SynchronizedSignals_Time.png'), 'Resolution', 300);

%% Figure 7: Trigger and early transient zoom
figure('Color','w');
plot(1000*t, currentOnGrid, 'LineWidth', 1.5); hold on;
plot(1000*t, coilForce, 'LineWidth', 1.5);
plot(1000*t, dispOnGrid, 'LineWidth', 1.5);
grid on; box on;
xlim([0 30]);
xlabel('Time (ms)'); ylabel('Signal value with original units');
title('Trigger and early transient region, 0 to 30 ms');
legend('Current (A)', 'Coil force (N)', 'Displacement (mm)', 'Location', 'best');
exportgraphics(gcf, fullfile(figFolder, 'Fig07_TriggerZoom_0_30ms.png'), 'Resolution', 300);

%% Figure 8: Force versus current
figure('Color','w');
plot(currentOnGrid, coilForce, 'LineWidth', 1.8);
grid on; box on;
xlabel('Coil current (A)'); ylabel('Coil force (N)');
title('Coil force versus input current');
exportgraphics(gcf, fullfile(figFolder, 'Fig08_Force_Current.png'), 'Resolution', 300);

%% Figure 9: Displacement versus current
figure('Color','w');
plot(currentOnGrid, dispOnGrid, 'LineWidth', 1.8);
grid on; box on;
xlabel('Coil current (A)'); ylabel('Z displacement (mm)');
title('Displacement versus input current');
exportgraphics(gcf, fullfile(figFolder, 'Fig09_Displacement_Current.png'), 'Resolution', 300);

%% Figure 10: Force versus displacement
figure('Color','w');
plot(dispOnGrid, coilForce, 'LineWidth', 1.8);
grid on; box on;
xlabel('Z displacement (mm)'); ylabel('Coil force (N)');
title('Force versus displacement');
exportgraphics(gcf, fullfile(figFolder, 'Fig10_Force_Displacement.png'), 'Resolution', 300);

%% Save cleaned data for the future LSTM step
cleanedData = table(t, currentOnGrid, coilForce, weightLoad, netForce, dispOnGrid, ...
    'VariableNames', {'Time_s','Current_A','CoilForce_N','WeightLoad_N','NetForce_N','Displacement_mm'});
writetable(cleanedData, 'ESA_cleaned_synchronized_data.csv');
save('ESA_cleaned_synchronized_data.mat', 'D', 'F', 'C', 'cleanedData');

disp('Data visualization completed successfully. Figures were saved in the figures folder.');

%% ============================================================
%  GitHub Push Block for LSTM_Modelling Repository
%  This block disconnects MATLAB from any previous repo path,
%  connects only to the correct repo, copies the current results,
%  commits, and pushes them to GitHub.
% ============================================================

disp('====================================================');
disp('Preparing GitHub push for LSTM_Modelling repository');
disp('====================================================');

%% GitHub settings
repoOwner = 'hzolfaghari2022';
repoName  = 'LSTM_Modelling';
branchName = 'main';
repoURLClean = ['https://github.com/' repoOwner '/' repoName '.git'];

% Folder where the current MATLAB script saved results.
sourceFolder = pwd;

% Temporary clean GitHub working folder.
githubRoot = fullfile(tempdir, 'MATLAB_GitHub_Repos');
repoFolder = fullfile(githubRoot, repoName);
targetFolderName = 'ESA_Data_Visualization_First_Report';
targetFolder = fullfile(repoFolder, targetFolderName);

% Remember current folder and clean up MATLAB path even if an error occurs.
originalFolder = pwd;
cleanupObj = onCleanup(@()safeReturnAndDisconnect(originalFolder, repoFolder));

%% Ask for GitHub token
% Important: do not save this token inside the script.
githubToken = strtrim(input('Paste your NEW GitHub Personal Access Token: ', 's'));
if isempty(githubToken)
    error('No GitHub token was entered. Create a token with write access and rerun the script.');
end

authRepoURL = ['https://' repoOwner ':' githubToken '@github.com/' repoOwner '/' repoName '.git'];

%% Step 1: Remove any previous repo folders from MATLAB path
fprintf('Removing previous GitHub repo folders from MATLAB path...\n');
if exist(githubRoot, 'dir')
    rmpath(genpath(githubRoot));
end

%% Step 2: Delete old local copy of this repo
if exist(repoFolder, 'dir')
    fprintf('Deleting old local copy of the repository...\n');
    [removeOK, removeMsg] = rmdir(repoFolder, 's');
    if ~removeOK
        error('Could not delete the old local repo folder:\n%s', removeMsg);
    end
end

if ~exist(githubRoot, 'dir')
    mkdir(githubRoot);
end

%% Step 3: Clone the correct repository
fprintf('Cloning the correct repository...\n');
cloneCommand = sprintf('git clone "%s" "%s"', authRepoURL, repoFolder);
[cloneStatus, cloneOut] = system(cloneCommand);

if cloneStatus ~= 0
    error(['Git clone failed. Possible reasons: wrong token, no repository access, ' ...
           'or the repository URL is incorrect.\n\nGit message:\n%s'], removeToken(cloneOut, githubToken));
end

% Add only this repo to MATLAB path after a successful clone.
addpath(genpath(repoFolder));

%% Step 4: Copy current project files into the repo
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

sourceFigFolder = fullfile(sourceFolder, 'figures');
targetFigFolder = fullfile(targetFolder, 'figures');
if exist(sourceFigFolder, 'dir')
    if exist(targetFigFolder, 'dir')
        rmdir(targetFigFolder, 's');
    end
    copyfile(sourceFigFolder, targetFigFolder);
end

%% Step 5: Create or update .gitignore
% Avoid pushing MATLAB Drive system files and temporary files.
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

%% Step 6: Git add, commit, and push
fprintf('Committing and pushing files to GitHub...\n');
cd(repoFolder);

% Keep the remote clean in the local repo, then use the token only for push.
system(sprintf('git remote set-url origin "%s"', repoURLClean));

% Make sure Git identity is configured for this temporary repo.
system('git config user.name "Hussein Zolfaghari"');
system('git config user.email "h.zolfaghari2015@gmail.com"');

% Force local branch name to main.
[branchStatus, branchOut] = system(['git branch -M ' branchName]);
if branchStatus ~= 0
    error('Could not set branch name to %s:\n%s', branchName, branchOut);
end

% Add all files.
[addStatus, addOut] = system('git add .');
if addStatus ~= 0
    error('Git add failed:\n%s', addOut);
end

% Show status after adding.
system('git status');

% Commit changes.
commitMessage = ['Update ESA data visualization and report - ' ...
                 datestr(now, 'yyyy-mm-dd HH:MM:SS')];
commitCommand = sprintf('git commit -m "%s"', commitMessage);
[commitStatus, commitOut] = system(commitCommand);

if commitStatus ~= 0
    if contains(commitOut, 'nothing to commit', 'IgnoreCase', true) || ...
       contains(commitOut, 'no changes added', 'IgnoreCase', true)
        fprintf('No new changes to commit. The repository may already be up to date.\n');
    else
        error('Git commit failed:\n%s', commitOut);
    end
else
    fprintf('Commit completed successfully.\n');
end

% Clear possible cached credentials, then push with the fresh token.
system('git credential-cache exit');
pushCommand = sprintf('git push -u "%s" %s', authRepoURL, branchName);
[pushStatus, pushOut] = system(pushCommand);

% Reset remote to the clean URL immediately after push attempt.
system(sprintf('git remote set-url origin "%s"', repoURLClean));

if pushStatus ~= 0
    safePushOut = removeToken(pushOut, githubToken);
    if contains(safePushOut, '403') || contains(safePushOut, 'Permission', 'IgnoreCase', true)
        error(['Git push failed because this token does not have write permission ' ...
               'for %s/%s.\n\nFix: revoke the exposed token, create a new token, ' ...
               'give it Contents: Read and write permission for this repository, ' ...
               'then rerun the script.\n\nGit message:\n%s'], ...
               repoOwner, repoName, safePushOut);
    else
        error('Git push to %s failed:\n%s', branchName, safePushOut);
    end
else
    fprintf('Files pushed successfully to GitHub %s branch.\n', branchName);
end

%% Step 7: Disconnect this repo path after push
fprintf('Removing this repository from MATLAB path after push...\n');
rmpath(genpath(repoFolder));
cd(sourceFolder);

disp('====================================================');
disp('GitHub push completed for LSTM_Modelling repository.');
disp('MATLAB is disconnected from this repository path.');
disp('====================================================');

%% Local functions
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

function cleanedText = removeToken(rawText, token)
    cleanedText = rawText;
    if ~isempty(token)
        cleanedText = strrep(cleanedText, token, '***TOKEN_REMOVED***');
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
