%% ============================================================
%  GitHub Auto Update Block for LSTM_Modelling
%  Put this block at the END of your MATLAB code.
%
%  What it does:
%  1. Saves your current project outputs into the GitHub repo folder.
%  2. Makes sure MATLAB is connected to the correct repository.
%  3. Adds, commits, and pushes the updated files to GitHub.
%  4. Prevents pushing to the wrong repository.
%
%  Repository:
%  https://github.com/hzolfaghari2022/LSTM_Modelling.git
% ============================================================

fprintf('\n====================================================\n');
fprintf('GitHub Auto Update for LSTM_Modelling\n');
fprintf('====================================================\n');

try
    %% ------------------------------------------------------------
    % User settings
    % -------------------------------------------------------------
    githubUser = 'hzolfaghari2022';
    repoName   = 'LSTM_Modelling';
    repoURL    = 'https://github.com/hzolfaghari2022/LSTM_Modelling.git';

    % This is the folder where your current MATLAB script is running.
    sourceFolder = pwd;

    % Local GitHub repository folder on your computer.
    % You can change this path if you want.
    githubRoot = fullfile(userpath, 'GitHub_Repos');
    githubRoot = erase(githubRoot, pathsep);

    repoFolder = fullfile(githubRoot, repoName);

    % Folder inside the GitHub repo where this project will be saved.
    targetFolder = fullfile(repoFolder, 'ESA_Data_Visualization_First_Report');

    %% ------------------------------------------------------------
    % Ask for GitHub token
    % -------------------------------------------------------------
    fprintf('\nPaste your NEW GitHub Personal Access Token.\n');
    fprintf('Important: the token must have Contents: Read and write permission.\n');
    githubToken = input('GitHub token: ', 's');

    if isempty(strtrim(githubToken))
        error('No GitHub token was entered. GitHub push was stopped.');
    end

    authRepoURL = ['https://' githubUser ':' githubToken ...
                   '@github.com/hzolfaghari2022/LSTM_Modelling.git'];

    %% ------------------------------------------------------------
    % Prepare local GitHub folder
    % -------------------------------------------------------------
    if ~exist(githubRoot, 'dir')
        mkdir(githubRoot);
    end

    %% ------------------------------------------------------------
    % Remove previous GitHub repo folders from MATLAB path
    % -------------------------------------------------------------
    fprintf('\nRemoving previous GitHub repo folders from MATLAB path...\n');

    if exist(githubRoot, 'dir')
        oldPaths = genpath(githubRoot);
        if ~isempty(oldPaths)
            rmpath(oldPaths);
        end
    end

    %% ------------------------------------------------------------
    % Clone repo if it does not exist yet
    % -------------------------------------------------------------
    if ~exist(repoFolder, 'dir')
        fprintf('Local repository does not exist. Cloning now...\n');

        cloneCommand = sprintf('git clone "%s" "%s"', authRepoURL, repoFolder);
        [cloneStatus, cloneOut] = system(cloneCommand);

        if cloneStatus ~= 0
            error('Git clone failed:\n%s', cloneOut);
        end

    else
        fprintf('Local repository already exists.\n');
    end

    %% ------------------------------------------------------------
    % Enter repository folder
    % -------------------------------------------------------------
    cd(repoFolder);

    %% ------------------------------------------------------------
    % Make sure this folder is a Git repo
    % -------------------------------------------------------------
    [insideStatus, insideOut] = system('git rev-parse --is-inside-work-tree');

    if insideStatus ~= 0 || ~contains(strtrim(insideOut), 'true')
        fprintf('Existing folder is not a valid Git repository.\n');
        fprintf('Deleting and cloning again...\n');

        cd(sourceFolder);
        rmdir(repoFolder, 's');

        cloneCommand = sprintf('git clone "%s" "%s"', authRepoURL, repoFolder);
        [cloneStatus, cloneOut] = system(cloneCommand);

        if cloneStatus ~= 0
            error('Git clone failed after reset:\n%s', cloneOut);
        end

        cd(repoFolder);
    end

    %% ------------------------------------------------------------
    % Make sure the repo remote is correct
    % -------------------------------------------------------------
    fprintf('Checking GitHub remote...\n');

    [remoteStatus, remoteOut] = system('git remote get-url origin');

    if remoteStatus ~= 0
        fprintf('No origin remote found. Adding origin...\n');
        system(sprintf('git remote add origin "%s"', repoURL));
    else
        fprintf('Current remote:\n%s\n', strtrim(remoteOut));

        if ~contains(remoteOut, 'github.com/hzolfaghari2022/LSTM_Modelling.git')
            error(['This folder is connected to a different repository. ' ...
                   'Stopping to avoid pushing to the wrong repo.']);
        end
    end

    %% ------------------------------------------------------------
    % Use authenticated remote temporarily for push
    % -------------------------------------------------------------
    system(sprintf('git remote set-url origin "%s"', authRepoURL));

    %% ------------------------------------------------------------
    % Add repo to MATLAB path temporarily
    % -------------------------------------------------------------
    addpath(genpath(repoFolder));

    %% ------------------------------------------------------------
    % Pull latest version first
    % -------------------------------------------------------------
    fprintf('Pulling latest changes from GitHub...\n');

    system('git branch -M main');

    [pullStatus, pullOut] = system('git pull origin main --rebase');

    if pullStatus ~= 0
        fprintf('Warning: git pull had an issue:\n%s\n', pullOut);
        fprintf('Continuing because this may be a new or empty repository.\n');
    end

    %% ------------------------------------------------------------
    % Create target folder inside repo
    % -------------------------------------------------------------
    if ~exist(targetFolder, 'dir')
        mkdir(targetFolder);
    end

    %% ------------------------------------------------------------
    % Copy current project files into the GitHub repo
    % -------------------------------------------------------------
    fprintf('Copying project files into the GitHub repository...\n');

    % Copy MATLAB files
    mFiles = dir(fullfile(sourceFolder, '*.m'));
    for k = 1:length(mFiles)
        copyfile(fullfile(sourceFolder, mFiles(k).name), targetFolder);
    end

    % Copy Excel files
    xlsxFiles = dir(fullfile(sourceFolder, '*.xlsx'));
    for k = 1:length(xlsxFiles)
        copyfile(fullfile(sourceFolder, xlsxFiles(k).name), targetFolder);
    end

    % Copy CSV files
    csvFiles = dir(fullfile(sourceFolder, '*.csv'));
    for k = 1:length(csvFiles)
        copyfile(fullfile(sourceFolder, csvFiles(k).name), targetFolder);
    end

    % Copy MAT files
    matFiles = dir(fullfile(sourceFolder, '*.mat'));
    for k = 1:length(matFiles)
        copyfile(fullfile(sourceFolder, matFiles(k).name), targetFolder);
    end

    % Copy TEX files
    texFiles = dir(fullfile(sourceFolder, '*.tex'));
    for k = 1:length(texFiles)
        copyfile(fullfile(sourceFolder, texFiles(k).name), targetFolder);
    end

    % Copy PDF files
    pdfFiles = dir(fullfile(sourceFolder, '*.pdf'));
    for k = 1:length(pdfFiles)
        copyfile(fullfile(sourceFolder, pdfFiles(k).name), targetFolder);
    end

    % Copy figures folder
    sourceFigFolder = fullfile(sourceFolder, 'figures');
    targetFigFolder = fullfile(targetFolder, 'figures');

    if exist(sourceFigFolder, 'dir')
        if exist(targetFigFolder, 'dir')
            rmdir(targetFigFolder, 's');
        end
        copyfile(sourceFigFolder, targetFigFolder);
    end

    %% ------------------------------------------------------------
    % Create or update .gitignore
    % -------------------------------------------------------------
    fprintf('Updating .gitignore...\n');

    gitignoreFile = fullfile(repoFolder, '.gitignore');

    ignoreLines = {
        ''
        '# MATLAB system files'
        '.MATLABDriveTag'
        '**/.MATLABDriveTag'
        '*.asv'
        'slprj/'
        ''
        '# Temporary files'
        '*.tmp'
        '*.log'
        ''
    };

    fid = fopen(gitignoreFile, 'a');

    if fid ~= -1
        for k = 1:length(ignoreLines)
            fprintf(fid, '%s\n', ignoreLines{k});
        end
        fclose(fid);
    else
        fprintf('Warning: could not update .gitignore.\n');
    end

    %% ------------------------------------------------------------
    % Configure Git identity
    % -------------------------------------------------------------
    system('git config user.name "Hussein Zolfaghari"');
    system('git config user.email "h.zolfaghari2015@gmail.com"');

    %% ------------------------------------------------------------
    % Add files
    % -------------------------------------------------------------
    fprintf('Adding files to Git...\n');

    [addStatus, addOut] = system('git add .');

    if addStatus ~= 0
        error('Git add failed:\n%s', addOut);
    end

    %% ------------------------------------------------------------
    % Check if there are changes to commit
    % -------------------------------------------------------------
    [diffStatus, ~] = system('git diff --cached --quiet');

    if diffStatus == 0
        fprintf('No new changes to commit.\n');
    else
        %% --------------------------------------------------------
        % Commit
        % ---------------------------------------------------------
        commitMessage = sprintf('Auto commit: ESA data visualization run %s', ...
            datestr(now, 'yyyy-mm-dd_HH-MM-SS'));

        commitCommand = sprintf('git commit -m "%s"', commitMessage);

        fprintf('Creating commit...\n');

        [commitStatus, commitOut] = system(commitCommand);

        if commitStatus ~= 0
            fprintf('Git commit failed or nothing changed:\n%s\n', commitOut);
        else
            fprintf('Commit created successfully:\n%s\n', commitMessage);
        end
    end

    %% ------------------------------------------------------------
    % Push to GitHub
    % -------------------------------------------------------------
    fprintf('Pushing to GitHub main branch...\n');

    [pushStatus, pushOut] = system('git push -u origin main');

    if pushStatus ~= 0
        fprintf('\nGit push failed.\n');
        fprintf('Git message:\n%s\n', pushOut);

        if contains(pushOut, '403') || contains(pushOut, 'Permission')
            fprintf('\nReason: your token does not have write permission.\n');
            fprintf('Fix: create a new GitHub token with Contents: Read and write permission for LSTM_Modelling.\n');
        end

        error('GitHub push did not complete.');
    else
        fprintf('Pushed successfully to GitHub.\n');
    end

    %% ------------------------------------------------------------
    % Reset remote URL to clean public URL so token is not stored
    % -------------------------------------------------------------
    system(sprintf('git remote set-url origin "%s"', repoURL));

    %% ------------------------------------------------------------
    % Disconnect repo from MATLAB path
    % -------------------------------------------------------------
    fprintf('Removing repository from MATLAB path...\n');

    rmpath(genpath(repoFolder));

    %% ------------------------------------------------------------
    % Return to original working folder
    % -------------------------------------------------------------
    cd(sourceFolder);

    fprintf('\n====================================================\n');
    fprintf('GitHub Auto Update Completed Successfully\n');
    fprintf('Repository updated:\n%s\n', repoURL);
    fprintf('====================================================\n');

catch ME
    %% ------------------------------------------------------------
    % Emergency cleanup
    % -------------------------------------------------------------
    try
        if exist('repoFolder', 'var') && exist(repoFolder, 'dir')
            cd(repoFolder);
            if exist('repoURL', 'var')
                system(sprintf('git remote set-url origin "%s"', repoURL));
            end
        end
    catch
    end

    try
        if exist('repoFolder', 'var') && exist(repoFolder, 'dir')
            rmpath(genpath(repoFolder));
        end
    catch
    end

    try
        if exist('sourceFolder', 'var') && exist(sourceFolder, 'dir')
            cd(sourceFolder);
        end
    catch
    end

    fprintf('\n====================================================\n');
    fprintf('GitHub Auto Update Failed\n');
    fprintf('Reason:\n%s\n', ME.message);
    fprintf('====================================================\n');
end