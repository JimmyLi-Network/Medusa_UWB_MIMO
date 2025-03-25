clear all;
close all
setsource;
rawDataFile = fullfile(rawBaseDir, rawRadarFile);

disp(sprintf('Loading %s', rawDataFile));
d = load(rawDataFile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate the radar cube
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The format of the radar data is (rangebins, frames, rxId, txId)
%radarData = d.radar_raw_data;
radarData = d.raw_radar_data;
[numRangeBins, numFrames, numRx, numTx] = size(radarData);
%radarData = radarData .* blackman(numRangeBins);
assert(~any(isnan(abs(radarData)), 'all'), 'NaNs found in input radar data');
%assert(numRx == numTx);

numElevationBins = numRx;
numAzimuthBins = numRx;
radarCube = zeros(numElevationBins, numAzimuthBins, numRangeBins, numFrames);

disp('Generating radar cube');
for frameId = 1:numFrames
    for rangeId = 1:numRangeBins
        v = squeeze(radarData(rangeId, frameId, :, :));
        d = fft2(v);
        %radarCube(:,:,rangeId, frameId) = rot90(d, 1);

        %dd = circshift(d, [numRx/2 numRx/2]);
        %radarCube(:,:,rangeId, frameId) = rot90(dd, 1);
        %radarCube(:,:,rangeId, frameId) = fftshift(fftshift(fft2(v), 1), 2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the heatmap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% imagesc(squeeze(abs(radarData(:,:,12,1))));
% colorbar;
% title('Signal heatmap');
% xlabel('Frames');
% ylabel('Range bins');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the group of bins that have the highest signal energy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% winSize = 20;
% numWindows = numRangeBins - winSize;
% winMag = zeros(1,numWindows);
% for i = 1:numWindows
%     m = sum(abs(radarData(i:(i+winSize-1), :,:,:)), 'all');
%     winMag(i) = m;
% end
% 
% [M,I] = max(winMag);
% S = radarData(I:(I+winSize-1), :,:,:);

winSize = 30;
numWindows = numRangeBins - winSize;
IndexArray = []
for numFrame = 1:4801
    for Num1 = 1:16
      for Num2 = 1:4
      winMag = zeros(1,numWindows);
      for i = 1:numWindows
          m = sum(abs(radarData(i:(i+winSize-1), numFrame,Num1,Num2)), 'all');
          winMag(i) = m;
      end

      [M,I] = max(winMag);
      IndexArray(numFrame,:) = I;
      maxBins = radarData(I:(I+winSize-1),numFrame,Num1,Num2);
      S(:,numFrame,Num1,Num2) = maxBins;
      end
    end 
end
 
% winSize = 10;
% for numFrame = 1:4801
%     for Num1 = 1:4
%         for Num2 = 1:16
%         [M,I] = maxk(radarData(:,numFrame,Num2,Num1), winSize);
%         maxBins = radarData(I,numFrame,Num2,Num1);
%         S(:,numFrame,Num2,Num1) = maxBins;
%         end
%     end
% end

% Plot the distribution
figure;
V = squeeze(sum(abs(S), [1 2]));
imagesc(V);
colorbar;
title('Signal strength across selected channels');

figure;
imagesc(squeeze(sum(abs(radarData), [1 2])));
colorbar
title('Signal strength across all channels');

channelFile = fullfile(rawBaseDir, 'channels.h5');
if exist(channelFile, 'file')
    delete(channelFile)
end

channelMatFile = fullfile(rawBaseDir, 'channels.mat');
if exist(channelMatFile, 'file')
    delete(channelMatFile)
end

%S(:,:,:,1) = S;S(:,:,:,2) = S;
S = reshape(S, [30,4801,16,4]); %[samples,frames,tx,rx]

% Create the HDF5 file
h5create(channelFile, '/data/I', size(S));
h5write(channelFile, '/data/I', real(S));
h5create(channelFile, '/data/Q', size(S));
h5write(channelFile, '/data/Q', imag(S));
h5create(channelFile, '/window/size', size(winSize));
h5write(channelFile, '/window/size', winSize);
h5create(channelFile, '/window/index', size(I));
h5write(channelFile, '/window/index', I);

% Write a .mat version for easier MATLAB use
winIndex = I;
channels = S;
save(channelMatFile, 'winSize', 'winIndex', 'channels', '-v7.3');

figure;
%imagesc(squeeze(abs(S(:,:,12,1))));
colorbar;
title(sprintf('Channel heatmap - Index %d', I));
xlabel('Frames');
ylabel('Range bins');
