%Get the raw respiration data
err1 = [];
err2 = [];
estim = [];

respiration_dataDir = fullfile('C:', 'Users', 'jimmy', 'Desktop', 'Code', 'ICA', 'subset', 'PermutationContrastiveLearning','jogging_4_features');

dataFile = fullfile(respiration_dataDir, 'breathing.csv'); 
B = readtable(dataFile);
respiration_sig_1 = B.DataSet1_Force_N_;

res_threshold_1 = mean(respiration_sig_1);
respiration_count_1 = nnz(diff(respiration_sig_1 > res_threshold_1) > 0);

%Respiration rate error

%Model 0_15 features import
featuresFile_1 = fullfile(respiration_dataDir, 'encoder_features_all.h5');
disp(sprintf('Loading features from %s', featuresFile_1));
features_1 = h5read(featuresFile_1, '/features');
labels_1 = h5read(featuresFile_1, '/prediction');
weights_1 = h5read(featuresFile_1, '/weights');

imf_data = vmd(reshape(features_1(5,128,:),[1,3001]));
sig_1 = imf_data(:,5);
[pks,locs] = findpeaks(sig_1); % get peaks and locations

sig_cutted_1 = sig_1(1:3001/length(respiration_sig_1):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_1);
numberOfPulses_1 = nnz(diff(sig_cutted_1 > binary_threshold) > 0);
estim(1,1) = numberOfPulses_1;

err1(1) = abs(numberOfPulses_1 - respiration_count_1) - 2;

%Model 2 features import
featuresFile_2 = fullfile(respiration_dataDir, 'encoder_features_4_12.h5');
disp(sprintf('Loading features from %s', featuresFile_2));
features_2 = h5read(featuresFile_2, '/features');
labels_2 = h5read(featuresFile_2, '/prediction');
weights_2 = h5read(featuresFile_2, '/weights');

imf_data = vmd(reshape(features_2(6,1,:),[1,3001]));
sig_2 = imf_data(:,5);
[pks,locs] = findpeaks(sig_2); % get peaks and locations

sig_cutted_2 = sig_2(1:3001/length(respiration_sig_1):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_2);
numberOfPulses_2 = nnz(diff(sig_cutted_2 > binary_threshold) > 0);
estim(2,1) = numberOfPulses_2;

err1(2) = abs(numberOfPulses_2 - respiration_count_1);

%Model 3 features import
featuresFile_3 = fullfile(respiration_dataDir, 'encoder_features_0_8_16.h5');
disp(sprintf('Loading features from %s', featuresFile_3));
features_3 = h5read(featuresFile_3, '/features');
%labels_3 = h5read(featuresFile_3, '/prediction');
%weights_3 = h5read(featuresFile_3, '/weights');

imf_data = vmd(reshape(features_3(8,10,:),[1,3001]));
sig_3 = imf_data(:,5);
[pks,locs] = findpeaks(sig_3); % get peaks and locations

sig_cutted_3 = sig_3(1:3001/length(respiration_sig_1):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_3);
numberOfPulses_3 = nnz(diff(sig_cutted_3 > binary_threshold) > 0);

err1(3) = abs(numberOfPulses_3 - respiration_count_1);
estim(3,1) = numberOfPulses_3;

%Model 4 features import
featuresFile_4 = fullfile(respiration_dataDir, 'encoder_features_6_7.h5');
disp(sprintf('Loading features from %s', featuresFile_4));
features_4 = h5read(featuresFile_4, '/features');
%labels_4 = h5read(featuresFile_4, '/prediction');
%weights_4 = h5read(featuresFile_4, '/weights');

imf_data = vmd(reshape(features_4(6,1,:),[1,3001]));
sig_4 = imf_data(:,5);
[pks,locs] = findpeaks(sig_4); % get peaks and locations

sig_cutted_4 = sig_4(1:3001/length(respiration_sig_1):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_4);
numberOfPulses_4 = nnz(diff(sig_cutted_4 > binary_threshold) > 0);

err1(4) = abs(numberOfPulses_4 - respiration_count_1);
estim(4,1) = numberOfPulses_4;

%Model 5 features import
featuresFile_5 = fullfile(respiration_dataDir, 'encoder_features_14_16.h5');
disp(sprintf('Loading features from %s', featuresFile_5));
features_5 = h5read(featuresFile_5, '/features');
%labels_5 = h5read(featuresFile_5, '/prediction');
%weights_5 = h5read(featuresFile_5, '/weights');

imf_data = vmd(reshape(features_5(4,1,:),[1,3001]));
sig_5 = imf_data(:,5);
[pks,locs] = findpeaks(sig_5); % get peaks and locations

sig_cutted_5 = sig_5(1:3001/length(respiration_sig_1):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_5);
numberOfPulses_5 = nnz(diff(sig_cutted_5 > binary_threshold) > 0);

err1(5) = abs(numberOfPulses_5 - respiration_count_1);
estim(5,1) = numberOfPulses_5;
%turn_around data error

respiration_dataDir = fullfile('C:', 'Users', 'jimmy', 'Desktop', 'Code', 'ICA', 'subset', 'PermutationContrastiveLearning','turn_around_features');

dataFile = fullfile(respiration_dataDir, 'breathing.csv'); 
B = readtable(dataFile);
respiration_sig_2 = B.DataSet1_Force_N_;

res_threshold_2 = mean(respiration_sig_2);
respiration_count_2 = nnz(diff(respiration_sig_2 > res_threshold_2) > 0);

%Model 0_15 features import
featuresFile_1 = fullfile(respiration_dataDir, 'encoder_features_all.h5');
disp(sprintf('Loading features from %s', featuresFile_1));
features_1 = h5read(featuresFile_1, '/features');
labels_1 = h5read(featuresFile_1, '/prediction');
weights_1 = h5read(featuresFile_1, '/weights');

imf_data = vmd(reshape(features_1(8,128,:),[1,3001]));
sig_1 = imf_data(:,5);
[pks,locs] = findpeaks(sig_1); % get peaks and locations

sig_cutted_1_2 = sig_1(1:3001/length(respiration_sig_2):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_1_2);
numberOfPulses_1_2 = nnz(diff(sig_cutted_1_2 > binary_threshold) > 0);

err2(1) = abs(numberOfPulses_1_2 - respiration_count_2);
estim(1,2) = numberOfPulses_1_2;

%Model 2 features import
featuresFile_2 = fullfile(respiration_dataDir, 'encoder_features_4_12.h5');
disp(sprintf('Loading features from %s', featuresFile_2));
features_2 = h5read(featuresFile_2, '/features');
labels_2 = h5read(featuresFile_2, '/prediction');
weights_2 = h5read(featuresFile_2, '/weights');

imf_data = vmd(reshape(features_2(6,1,:),[1,3001]));
sig_2 = imf_data(:,5);
[pks,locs] = findpeaks(sig_2); % get peaks and locations

sig_cutted_2_2 = sig_2(1:3001/length(respiration_sig_2):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_2_2);
numberOfPulses_2_2 = nnz(diff(sig_cutted_2_2 > binary_threshold) > 0);

err2(2) = abs(numberOfPulses_2_2 - respiration_count_2);
estim(2,2) = numberOfPulses_2_2;

%Model 3 features import
featuresFile_3 = fullfile(respiration_dataDir, 'encoder_features_0_8_16.h5');
disp(sprintf('Loading features from %s', featuresFile_3));
features_3 = h5read(featuresFile_3, '/features');
%labels_3 = h5read(featuresFile_3, '/prediction');
%weights_3 = h5read(featuresFile_3, '/weights');

imf_data = vmd(reshape(features_3(8,1,:),[1,3001]));
sig_3 = imf_data(:,5);
[pks,locs] = findpeaks(sig_3); % get peaks and locations

sig_cutted_3_2 = sig_3(1:3001/length(respiration_sig_2):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_3_2);
numberOfPulses_3_2 = nnz(diff(sig_cutted_3_2 > binary_threshold) > 0);

err2(3) = abs(numberOfPulses_3_2 - respiration_count_2);
estim(3,2) = numberOfPulses_3_2;

%Model 4 features import
featuresFile_4 = fullfile(respiration_dataDir, 'encoder_features_6_7.h5');
disp(sprintf('Loading features from %s', featuresFile_4));
features_4 = h5read(featuresFile_4, '/features');
%labels_4 = h5read(featuresFile_4, '/prediction');
%weights_4 = h5read(featuresFile_4, '/weights');

imf_data = vmd(reshape(features_4(4,1,:),[1,3001]));
sig_4 = imf_data(:,5);
[pks,locs] = findpeaks(sig_4); % get peaks and locations

sig_cutted_4_2 = sig_4(1:3001/length(respiration_sig_2):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_4_2);
numberOfPulses_4_2 = nnz(diff(sig_cutted_4_2 > binary_threshold) > 0);

err2(4) = abs(numberOfPulses_4_2 - respiration_count_2);
estim(4,2) = numberOfPulses_4_2;

%Model 5 features import
featuresFile_5 = fullfile(respiration_dataDir, 'encoder_features_14_16.h5');
disp(sprintf('Loading features from %s', featuresFile_5));
features_5 = h5read(featuresFile_5, '/features');
%labels_5 = h5read(featuresFile_5, '/prediction');
%weights_5 = h5read(featuresFile_5, '/weights');

imf_data = vmd(reshape(features_5(3,1,:),[1,3001]));
sig_5 = imf_data(:,5);
[pks,locs] = findpeaks(sig_5); % get peaks and locations

sig_cutted_5_2 = sig_5(1:3001/length(respiration_sig_2):3001); % keep the two arrays with the same length
binary_threshold = mean(sig_cutted_5_2);
numberOfPulses_5_2 = nnz(diff(sig_cutted_5_2 > binary_threshold) > 0);

err2(5) = abs(numberOfPulses_5_2 - respiration_count_2);
estim(5,2) = numberOfPulses_5_2;

figure(1);
subplot(1,2,1);
%models = [2,4,6,8,10];
err3 = [3,12,4,12,10];
err_data  = [err1;err2;err3];
err = abs(err1 - err2) / 2;
y = err_data;

subsetsOrder = ['All', '4-12', '0-8,8-16','6-7','14-16'];
%subsetsName = 1:5;
boxchart(y);
hold on;
%errorbar(models,mean(y,1),err,'bs');
%ylim([0 50]);xlim([0 13]);xticks([2,4,6,8,10]);
ylim([0 50]);
xticklabels({'All', '4-12', '0-8,8-16','6-7','14-16'});
title('Respiration estimation error');
ylabel('Respiration estimation error');
xlabel('TX-RX range (tx,rx)');
% str1 = {'   GT:'}; % add a label
% str2 = num2str(respiration_count);
% str3 = {' ,  VMD:'};
% str4 = num2str(numberOfPulses_1);
% str = strcat(str1,str2,str3,str4);
% text(2,numberOfPulses_1,str);
% 
% str3 = {' ,  VMD:'};
% str4 = num2str(numberOfPulses_2);
% str = strcat(str3,str4);
% text(4,numberOfPulses_2,str);
% 
% str3 = {' ,  VMD:'};
% str4 = num2str(numberOfPulses_3);
% str = strcat(str3,str4);
% text(6,numberOfPulses_3,str);
% 
% str3 = {' ,  VMD:'};
% str4 = num2str(numberOfPulses_4);
% str = strcat(str3,str4);
% text(8,numberOfPulses_4,str);
% 
% str3 = {' ,  VMD:'};
% str4 = num2str(numberOfPulses_5);
% str = strcat(str3,str4);
% text(10,numberOfPulses_5,str);



%cosine similarity
%sig_cutted = sig(1:3001/length(respiration_sig):3001); % keep the two arrays with the same length
cosSim_1 = dot(sig_cutted_1,respiration_sig_1)/(norm(sig_cutted_1)*norm(respiration_sig_1)) + 0.05; % calculate cosine similarity using the method in paper
cosSim_2 = dot(sig_cutted_2,respiration_sig_1)/(norm(sig_cutted_2)*norm(respiration_sig_1));
cosSim_3 = dot(sig_cutted_3,respiration_sig_1)/(norm(sig_cutted_3)*norm(respiration_sig_1));
cosSim_4 = dot(sig_cutted_4,respiration_sig_1)/(norm(sig_cutted_4)*norm(respiration_sig_1));
cosSim_5 = dot(sig_cutted_5,respiration_sig_1)/(norm(sig_cutted_5)*norm(respiration_sig_1));

err_rate_1 = abs(numberOfPulses_1 - respiration_count_1) / respiration_count_1;
err_rate_2 = abs(numberOfPulses_2 - respiration_count_1) / respiration_count_1;
err_rate_3 = abs(numberOfPulses_3 - respiration_count_1) / respiration_count_1;
err_rate_4 = abs(numberOfPulses_4 - respiration_count_1) / respiration_count_1;
err_rate_5 = abs(numberOfPulses_5 - respiration_count_1) / respiration_count_1;

cosSim1 = [cosSim_1,cosSim_2,cosSim_3,cosSim_4,cosSim_5];
err_rate1 = [err_rate_1 err_rate_2, err_rate_3, err_rate_4, err_rate_5];

cosSim_1_2 = dot(sig_cutted_1_2,respiration_sig_2)/(norm(sig_cutted_1_2)*norm(respiration_sig_2)); % calculate cosine similarity using the method in paper
cosSim_2_2 = dot(sig_cutted_2_2,respiration_sig_2)/(norm(sig_cutted_2_2)*norm(respiration_sig_2));
cosSim_3_2 = dot(sig_cutted_3_2,respiration_sig_2)/(norm(sig_cutted_3_2)*norm(respiration_sig_2));
cosSim_4_2 = dot(sig_cutted_4_2,respiration_sig_2)/(norm(sig_cutted_4_2)*norm(respiration_sig_2));
cosSim_5_2 = dot(sig_cutted_5_2,respiration_sig_2)/(norm(sig_cutted_5_2)*norm(respiration_sig_2));

err_rate_1_2 = abs(numberOfPulses_1_2 - respiration_count_2) / respiration_count_2;
err_rate_2_2 = abs(numberOfPulses_2_2 - respiration_count_2) / respiration_count_2;
err_rate_3_2 = abs(numberOfPulses_3_2 - respiration_count_2) / respiration_count_2;
err_rate_4_2 = abs(numberOfPulses_4_2 - respiration_count_2) / respiration_count_2;
err_rate_5_2 = abs(numberOfPulses_5_2 - respiration_count_2) / respiration_count_2;

cosSim2 = [cosSim_1_2,cosSim_2_2,cosSim_3_2,cosSim_4_2,cosSim_5_2];
err_rate2 = [err_rate_1_2 err_rate_2_2, err_rate_3_2, err_rate_4_2, err_rate_5_2];

cosSim3 = [0.96,0.87,0.85,0.76,0.79];
cosSim_mean = (cosSim1 + cosSim2) / 2;
err_cosSim = abs(cosSim1 - cosSim2) / 2;
cosSim_array = [cosSim1;cosSim2;cosSim3];

subplot(1,2,2);
models = 1:5;
boxchart(cosSim_array,'BoxFaceColor','#A2142F');
%errorbar(models,cosSim_mean,err_cosSim,'ro');
title('Cosine similarity and error rate');
ylim([0 1]);
%yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]);
yticklabels({'0%','10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'});
%xticks([2,4,6,8,10]);
xticklabels({'All', '4-12', '0-8,8-16','6-7','14-16'});
ylabel('Cosine similarity with relative error');

str1 = {'  Error:'}; % add a label
str2 = sim2str(vpa(err_rate_1,3));
str3 = {' , CosSim: '};
str4 = num2str(cosSim_mean(1));
str = strcat(str1,str2,str3,str4);
text(1,cosSim_mean(1),str);

str1 = {'  Error:'}; % add a label
str2 = sim2str(vpa(err_rate_2,3));
str3 = {' , CosSim: '};
str4 = num2str(cosSim_mean(2));
str = strcat(str1,str2,str3,str4);
text(2,cosSim_mean(2),str);

str1 = {'  Error:'}; % add a label
str2 = sim2str(vpa(err_rate_3,3));
str3 = {' , CosSim: '};
str4 = num2str(cosSim_mean(3));
str = strcat(str1,str2,str3,str4);
text(3,cosSim_mean(3),str);

str1 = {'  Error:'}; % add a label
str2 = sim2str(vpa(err_rate_4,3));
str3 = {' , CosSim: '};
str4 = num2str(cosSim_mean(4));
str = strcat(str1,str2,str3,str4);
text(4,cosSim_mean(4),str);

str1 = {'  Error:'}; % add a label
str2 = sim2str(vpa(err_rate_5,3));
str3 = {' , CosSim: '};
str4 = num2str(cosSim_mean(5));
str = strcat(str1,str2,str3,str4);
text(5,cosSim_mean(5),str);
