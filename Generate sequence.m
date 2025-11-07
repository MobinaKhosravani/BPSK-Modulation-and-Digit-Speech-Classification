clc;
clear;
close all;

%% Load Audio Files into MATLAB
disp('Loading audio files...');
datasetPath = 'D:\Uni\singals and systems\Dataset\free-spoken-digit-dataset-master\recordings';
audioFiles = dir(fullfile(datasetPath, '*.wav'));

disp(['Total audio files found: ', num2str(length(audioFiles))]);

% Create a folder to save the generated random sequences
outputFolder = 'D:\Uni\GeneratedSequences';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Number of sequences to generate
numSequences = 9;

% Maximum length of each sequence
maxSequenceLength = 4;

% Generate and save sequences
for seqIdx = 1:numSequences
    % Randomly pick a sequence length between 3 and maxSequenceLength
    seqLength = randi([3, maxSequenceLength]);
    
    sequenceAudioData = [];
    sequenceDigits = '';  % To store the digit sequence
    
    for i = 1:seqLength
        % Randomly pick an audio file from the dataset
        randomIndex = randi(length(audioFiles));
        filePath = fullfile(audioFiles(randomIndex).folder, audioFiles(randomIndex).name);
        [audioData, fs] = audioread(filePath);
        
        % Convert to mono if stereo
        if size(audioData, 2) > 1
            audioData = mean(audioData, 2);
        end
        
        % Append audio data to form the sequence
        sequenceAudioData = [sequenceAudioData; audioData];
        
        % Extract the digit from the file name and append it to the sequenceDigits string
        digitLabel = str2double(regexp(audioFiles(randomIndex).name, '\d', 'match', 'once'));
        sequenceDigits = [sequenceDigits, num2str(digitLabel)];
    end
    
    % Save the generated sequence as a .wav file
    sequenceFileName = fullfile(outputFolder, [sequenceDigits, '.wav']);
    audiowrite(sequenceFileName, sequenceAudioData, fs);
    disp(['Saved sequence ', sequenceDigits]);
end
