clear all
close all

% Load the data cubes
% c_i = h5read('data/radar_cube_I_1.h5', '/source');
% c_q = h5read('data/radar_cube_Q_1.h5', '/source');
% c = c_i + sqrt(-1)*c_q;
%c = load('~/Downloads/radar_cube_array_101.mat', 'radar_cube_array');
c = load('data/1tx16rx/radar_processed_AoA.mat');

% The radar cube array has the dimensions (range, azimuth, frame_id)
cm = abs(c.fft2d_frames);

%[maxval, idx] = max(cm, [], 3);

cfar = phased.CFARDetector2D('GuardBandSize', 5, 'TrainingBandSize', 5, ...
    'ThresholdFactor', 'Auto', ...
    'ProbabilityFalseAlarm', 0.2)
gap = floor(1.5*(cfar.GuardBandSize + cfar.TrainingBandSize));
[rowInd, colInd] = meshgrid(gap:(size(cm, 1) - gap),...
     gap:(size(cm, 2) - gap));
cutInd = [rowInd(:) colInd(:)]';

for f = 1:size(cm, 3)
    v = squeeze(cm(:,:,f));
    imagesc(v);

    det = cfar(v, cutInd);
    coord = cutInd(:, det);
    idx = dbscan(coord', 2, 5);
    
    hold on;
    gscatter(coord(2,:), coord(1,:), idx);
    hold off;

    title(sprintf('Frame %d', f));
    xlabel('Azimuth');
    ylabel('Elevation');
    colorbar;
    drawnow;
    pause(0.5);
end