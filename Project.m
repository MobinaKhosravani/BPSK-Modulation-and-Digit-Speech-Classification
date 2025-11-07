clc;
clear;
close all;

%% Load Audio Files into MATLAB
disp('Loading audio files...');
datasetPath = 'D:\Uni\singals and systems\Dataset\free-spoken-digit-dataset-master\recordings';
audioFiles = dir(fullfile(datasetPath, '*.wav'));

disp(['Total audio files found: ', num2str(length(audioFiles))]);


features = [];
labels = [];


numCoeffs = 13;
trainRatio = 0.8;
numFolds = 10;


disp('Extracting features from audio files...');
for i = 1:length(audioFiles)

    filePath = fullfile(audioFiles(i).folder, audioFiles(i).name);
    [audioData, fs] = audioread(filePath);
    
    
    if size(audioData, 2) > 1
        audioData = mean(audioData, 2);  
    end
    
    feature_vector = extractFeatures(audioData, fs, numCoeffs);
    

    digitLabel = str2double(regexp(audioFiles(i).name, '\d', 'match', 'once'));
    
   
    features = [features; feature_vector];
    labels = [labels; digitLabel];
end
disp('Feature extraction complete.');


%% Train-Test Split

disp('Splitting dataset into training and test sets...');
cv = cvpartition(labels, 'HoldOut', 1 - trainRatio);
X_train = features(training(cv), :);
Y_train = labels(training(cv));
X_test = features(test(cv), :);
Y_test = labels(test(cv));

%% Train Model

disp('Training models...');
svmModel = fitcecoc(X_train, Y_train);


%% Test Model
disp('Testing models...');
svmPredictions = predict(svmModel, X_test);


svmAccuracy = sum(svmPredictions == Y_test) / numel(Y_test);


disp(['SVM Test Accuracy: ', num2str(svmAccuracy * 100), '%']);


figure;
confusionchart(Y_test, svmPredictions);
title('SVM Confusion Matrix');


%% Test New Audio File
disp('Select an audio file for prediction...');
[file, path] = uigetfile('*.wav', 'Select an Audio File');
if file ~= 0
    [audioData, fs] = audioread(fullfile(path, file));
    if size(audioData, 2) > 1
        audioData = mean(audioData, 2);  
    end
    
    
    numCoeffs = 13;  
    
    feature_vector = extractFeatures(audioData, fs, numCoeffs);
    
  
    predictedSVM = predict(svmModel, feature_vector);
  
   
    disp(['SVM Prediction: ', num2str(predictedSVM)]);
else
    disp('No file selected.');
end


%% Predict a sequence of numbers

predictDigitsFromFolder(svmModel, 13);


%% Feature extraction function
function feature_vector = extractFeatures(audioData, fs, numCoeffs)
    % 1. FFT Features
    feature_fft = abs(fft(audioData));
    feature_fft = feature_fft(1:floor(length(feature_fft)/2)); % Take positive frequencies
    mean_fft = mean(feature_fft);

    % 2. STFT Features
    windowSize = 256; 
    overlap = 128;    
    [S, ~, ~] = stft(audioData, fs, 'Window', hamming(windowSize), 'OverlapLength', overlap);
    mean_stft = mean(abs(S(:)));

    % 3. Cepstrum Features
    logSpectrum = log(abs(fft(audioData)) + eps);
    cepstrum = real(ifft(logSpectrum));
    mean_cepstrum = mean(cepstrum(1:floor(length(cepstrum)/2)));  % Take the positive cepstrum

    % 4. MFCC Features (mean across frames)
    coeffs = mfcc(audioData, fs, 'NumCoeffs', numCoeffs);
    mean_mfcc = mean(coeffs);
    
    % 5. Spectral Features
    spectralCentroid = sum((1:length(feature_fft)) .* feature_fft') / sum(feature_fft);
    spectralBandwidth = sqrt(sum(((1:length(feature_fft)) - spectralCentroid).^2 .* feature_fft') / sum(feature_fft));
    
    % 6. Temporal Features
    zeroCrossRate = sum(abs(diff(audioData > 0))) / length(audioData);
    rmsEnergy = sqrt(mean(audioData.^2));
    
    % 7. Spectral Roll-off (percentage of energy in the lower frequencies)
    spectralRollOff = sum(feature_fft > 0.85 * max(feature_fft)) / length(feature_fft);
    
    % 8. Spectral Flatness (describes noisiness of the signal)
    spectralFlatness = geomean(feature_fft) / mean(feature_fft);

    
    % 9. Zero-Crossing Rate (normalized)
    zcr = zeroCrossingRate(audioData);  
    
    % Combine all features into one vector
    feature_vector = [mean_fft, mean_stft, mean_cepstrum, mean_mfcc, ...
                      spectralCentroid, spectralBandwidth, zeroCrossRate, ...
                      rmsEnergy, spectralRollOff, spectralFlatness, ...
                       zcr];
end


%% Zero-Crossing Rate function
function zcr = zeroCrossingRate(audioData)
    zcr = sum(abs(diff(audioData > 0))) / length(audioData);  % Zero-crossing rate
end

%%
% 
% function predictDigitsFromFile(svmModel, numCoeffs)
%     % Step 1: Allow the user to upload a file using a file dialog
%     [fileName, filePath] = uigetfile('*.wav', 'Select an Audio File');
%     
%     if fileName == 0
%         disp('No file selected. Exiting function.');
%         return;
%     end
%     
%     % Full path to the selected file
%     uploadedFilePath = fullfile(filePath, fileName);
%     
%     % Step 2: Load the selected audio file
%     [audioData, fs] = audioread(uploadedFilePath);
%     
%     % Step 3: Convert stereo to mono if necessary
%     if size(audioData, 2) > 1
%         audioData = mean(audioData, 2);  % Convert to mono by averaging channels
%     end
%     
%     % Step 4: Estimate segment length and number of digits in the sequence
%     % Assuming each digit lasts about 0.5 seconds on average
%     numDigits = floor(length(audioData) / (fs * 0.5)); % Approximate number of digits
%     segmentLength = floor(length(audioData) / numDigits);
%     
%     % Step 5: Initialize an empty string to store the predicted digits
%     predictedDigits = '';
%     
%     % Step 6: Segment the audio data into individual digits and predict each one
%     for i = 1:numDigits
%         % Calculate the start and end indices for the current segment
%         startIdx = (i - 1) * segmentLength + 1;
%         endIdx = i * segmentLength;
%         
%         % Extract the audio segment for the current digit
%         digitSegment = audioData(startIdx:endIdx);
%         
%         % Step 7: Extract features from the audio segment
%         feature_vector = extractFeatures(digitSegment, fs, numCoeffs);
%         
%         % Step 8: Use the SVM model to predict the digit
%         predictedDigit = predict(svmModel, feature_vector);
%         
%         % Append the predicted digit to the predicted digits string
%         predictedDigits = [predictedDigits, num2str(predictedDigit)];
%     end
%     
%     % Step 9: Display the predicted digits (the full number)
%     disp(['Predicted number: ', predictedDigits]);
% end

%%
function predictDigitsFromFolder(svmModel, numCoeffs)
    % Step 1: Set the path where the generated sequences are stored
    folderPath = 'D:\Uni\GeneratedSequences';  % Change this path if necessary
    
    % Get the list of all .wav files in the folder
    audioFiles = dir(fullfile(folderPath, '*.wav'));
    
    if isempty(audioFiles)
        disp('No audio files found in the folder.');
        return;
    end
    
    % Step 2: Loop over each audio file and predict the digits
    for fileIdx = 1:length(audioFiles)
        % Get the full path of the current audio file
        filePath = fullfile(audioFiles(fileIdx).folder, audioFiles(fileIdx).name);
        
        % Load the audio file
        [audioData, fs] = audioread(filePath);
        
        % Convert stereo to mono if necessary
        if size(audioData, 2) > 1
            audioData = mean(audioData, 2);  % Convert to mono by averaging channels
        end
        
        % Step 3: Estimate segment length and number of digits in the sequence
        numDigits = floor(length(audioData) / (fs * 0.5));  % Approximate number of digits (0.5s each)
        segmentLength = floor(length(audioData) / numDigits);
        
        % Initialize an empty string to store the predicted digits
        predictedDigits = '';
        
        % Segment the audio data into individual digits and predict each one
        for i = 1:numDigits
            % Calculate the start and end indices for the current segment
            startIdx = (i - 1) * segmentLength + 1;
            endIdx = i * segmentLength;
            
            % Extract the audio segment for the current digit
            digitSegment = audioData(startIdx:endIdx);
            
            % Extract features from the audio segment
            feature_vector = extractFeatures(digitSegment, fs, numCoeffs);
            
            % Use the SVM model to predict the digit
            predictedDigit = predict(svmModel, feature_vector);
            
            % Append the predicted digit to the predicted digits string
            predictedDigits = [predictedDigits, num2str(predictedDigit)];
        end
        
        % Step 4: Display the filename and the predicted number
        disp(['File: ', audioFiles(fileIdx).name, ' - Predicted Number: ', predictedDigits]);
    end
end