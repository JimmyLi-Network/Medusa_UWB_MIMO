
clear all;
close all;

% sourceDir = 'D:\data\uwb\Slow_Walking\raw_radar_data.mat';
% s = load(sourceDir);

setsource;

dataDir = rawBaseDir;
%featuresDir = fullfile(pwd, 'checkpoint'); 
featuresDir = pwd;
dataFile = fullfile(dataDir, 'breathing.csv');

if exist(dataFile, 'file')
    B = readtable(dataFile);
    figure;
    plot(B.DataSet1_Time_s_, B.DataSet1_Force_N_);
    title('Breathing force sensor');
    xlabel('Time (s)');
    grid on;
end

featuresFile = fullfile(featuresDir, 'encoder_features.h5');
disp(sprintf('Loading features from %s', featuresFile));
features = h5read(featuresFile, '/features');
labels = h5read(featuresFile, '/prediction');
weights = h5read(featuresFile, '/weights');

[M,I] = max(weights, [], 2);
F = zeros(size(features, [1 3]));
for i = 1:size(features,3)
    F(:,i) = squeeze(features(:,squeeze(I(:,:,i)), i));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Perform VMD across all the features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('VMD');
[numChan numSamples] = size(features);
imf = {};
for i = 1:numChan
    disp(sprintf('Channel %d', i));
    f = squeeze(F(i,:));
    y = vmd(f);
    imf{i} = y;
end

% disp('VMD');
% [numChan numSamples numUnits] = size(features);
% imf = {};
% for i = 1:numUnits
%     for j = 1:numChan
%         disp(sprintf('Unit: %d, Chan: %d', i, j));
%         f = squeeze(features(j,:,i));
%         y = vmd(f);
%         imf{i,j} = y;
%     end
% end

% Save the VMD output to file, since this takes a while to compute
vmdFile = fullfile(featuresDir, 'signalvmd.mat');
save(vmdFile, 'numChan', 'numSamples', 'imf', '-v7.3');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Plot the nonlinear ICA components
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% figure;
% plot(features');
% grid on
% xlabel('Samples');
% title('Features of radar signal');
% 
% figure;
% plot(labels);
% grid on
% xlabel('Samples');
% title('Labels of radar signal');
% 
% componentsFile = fullfile(featuresDir, 'feature_components.h5');
% if exist(componentsFile, 'file')
%     components = h5read(componentsFile, '/components');
%     figure
%     plot(components');
%     grid on;
%     xlabel('Samples')
%     title('Linear ICA components of radar signals');
% end


