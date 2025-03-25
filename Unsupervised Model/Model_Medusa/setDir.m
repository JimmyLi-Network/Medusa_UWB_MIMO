
featuresFolder = '\xinxin\thruwall';
%configure the directories of datasets
respiration_dataDir = fullfile('C:', 'Users', 'jimmy', 'Desktop', 'Code', 'ICA', 'subset', 'PermutationContrastiveLearning',featuresFolder);
rawDataDir = fullfile('C:', 'Users', 'jimmy', 'Desktop', 'Code', 'ICA', 'subset', 'PermutationContrastiveLearning', featuresFolder);

dataFile = fullfile(respiration_dataDir, 'breathing.csv'); 
featuresFile = 'encoder_features_6_7.h5';
featuresDir = fullfile(rawDataDir,featuresFile);
%rawRadarFile = 'raw_radar_data.mat';
rawRadarFile = '1TX_16RX.mat';
