% First processing step in the radar pipeline
% First step: showcube.m
% Second step: tracker.m
% Third step: findtracks.m

clear all
close all

% Load the data cubes
% c_i = h5read('data/radar_cube_I_1.h5', '/source');
% c_q = h5read('data/radar_cube_Q_1.h5', '/source');
% c = c_i + sqrt(-1)*c_q;
%c = load('~/Downloads/radar_cube_array_101.mat', 'radar_cube_array');
%c = load('data/16tx16rx/radar_cube_array_101.mat', 'radar_cube_array');
%

setsource;

videoFile = fullfile(baseDir, 'radar_cube.avi');
if exist(videoFile, 'file')
    delete(videoFile);
end
video = VideoWriter(videoFile);
video.FrameRate = 40;
open(video);


showPlots = true;
for arrayId = arrayRange
    inputFilename = fullfile(baseDir, sprintf('radar_cube_array_%d.mat', arrayId));
    outputFilename = fullfile(baseDir, sprintf('radar_cube_array_result_%d.mat', arrayId));

    disp(inputFilename);
    c = load(inputFilename, 'radar_cube_array');
    % shiftSteps = size(c.radar_cube_array,1)/2;
    % radarData = circshift(c.radar_cube_array,[shiftSteps, shiftSteps]);
    radarData = c.radar_cube_array;
    clusters = findRadarClusters(radarData, showPlots, video);

    if ~isempty(outputFilename)
        save(outputFilename, 'clusters', '-v7.3');
    end
end

close(video);

function clusters = findRadarClusters(radarData, showPlots, video)
    
    % The radar cube array has the dimensions (elevation, azimuth, range, frame_id)
    cm = abs(radarData);

    [maxVal, maxId] = max(cm, [], 3);
    meanVal = abs(mean(radarData, 3));

    cfar = phased.CFARDetector2D('GuardBandSize', 15, 'TrainingBandSize', 5, ...
        'ThresholdFactor', 'Auto', ...
        'ProbabilityFalseAlarm', 0.25)
    gap = floor(1.5*(cfar.GuardBandSize + cfar.TrainingBandSize));
    [rowInd, colInd] = meshgrid(gap:(size(cm, 1) - gap),...
         gap:(size(cm, 2) - gap));
    cutInd = [rowInd(:) colInd(:)]';

    clusters = [];
    %maxId = 0;
    for f = 1:size(cm, 4)
        v = squeeze(maxVal(:,:,1,f));
        range = squeeze(maxId(:,:,1,f));

        det = cfar(v, cutInd);
        coord = cutInd(:, det);
        idx = dbscan(coord', 2, 5);
        %data = radarData(coord(1,:), coord(2,:), :, f);

        clusterGroups = unique(idx);
        centroid = zeros(length(clusterGroups), 2);
        newCentroid = zeros(length(clusterGroups), 2);
        for k = 1:length(clusterGroups)
            g = clusterGroups(k);
            if g == -1
                continue
            end

            % Find the centroid of each detected cluster
            xm = mean(coord(2, idx==g));
            ym = mean(coord(1, idx==g));
            centroid(k, :) = [xm, ym];
        end

        %clusters(f).data = data;
        clusters(f).points = coord;
        clusters(f).indices = idx;
        clusters(f).groups = clusterGroups;
        clusters(f).centroids = centroid;

        % Plot the current view
        if showPlots
            imagesc(v);
            hold on;
            gscatter(coord(2,:), coord(1,:), idx);
            for i = 1:size(centroid, 1)
                text(centroid(i,1), centroid(i,2), sprintf('%d', clusterGroups(i)));
            end
            %scatter(centroid(:,1), centroid(:,2), 'x', 'HandleVisibility', 'off');
            hold off;
            title(sprintf('Frame %d', f));
            xlabel('Azimuth');
            ylabel('Elevation');
            colorbar;
            drawnow;

            frame = getframe(gcf);
            writeVideo(video, frame); 

            % pause(0.5);
        end
    end

    %resultFile = fullfile(baseDir, outputFilename);
    %save(resultFile, 'clusters', '-v7.3');
end