
% Second processing step in the radar pipeline
% First step: showcube.m
% Second step: tracker.m
% Third step: findtracks.m


clear all
close all

setsource;

motionTracker = MotionTracker;
frameId = 0;
for arrayId = arrayRange
    dataFilename = sprintf('radar_cube_array_%d.mat', arrayId);
    inputFilename = sprintf('radar_cube_array_result_%d.mat', arrayId);
    resultFile = fullfile(baseDir, inputFilename);
    dataFile = fullfile(baseDir, dataFilename);

    disp(resultFile);
    d = load(resultFile, 'clusters');
    clusters = d.clusters;

    % The radar cube array has the dimensions (elevation, azimuth, range, frame_id)
    d = load(dataFile);
    radarData = d.radar_cube_array;

    nextFrameId = trackRadarClusters(motionTracker, clusters, radarData, arrayId, frameId);
    frameId = nextFrameId;
end

motionTrackerFilename = fullfile(baseDir, 'motionTrackedResult.mat');
save(motionTrackerFilename, 'motionTracker', '-v7.3');

function nextFrameId = trackRadarClusters(motionTracker, clusters, radarData, resultId, firstFrameId)

    
    %motionTracker = MotionTracker;

    for i = 1:length(clusters)
        frameId = firstFrameId + i;


        % Update the tracking information
        currPoints = clusters(i).points;
        currGroups = clusters(i).groups;
        currCentroids = clusters(i).centroids;
        currIndices = clusters(i).indices;

        pointData = squeeze(radarData(:,:,:,i));

        motionTracker.addPoints(frameId, currPoints, pointData, currIndices);

        subplot(211);
        gscatter(currPoints(2,:), currPoints(1,:), currIndices);
        hold on;
        for j = 1:size(currCentroids, 1)
            text(currCentroids(j,1), currCentroids(j,2), sprintf('%d', currGroups(j)));
        end
        hold off;
        title(sprintf('Result %d', resultId));

        subplot(212);
        [points, data, trackIds] = motionTracker.getPoints(frameId);
        plot(0,0,'x');
        hold on;
        for pIdx = 1:length(points)
            p = points{pIdx};
            scatter(p(2,:), p(1,:));
            c = mean(p, 2);
            text(c(2), c(1), sprintf('%d', trackIds(pIdx)));
        end
        hold off;

        title(sprintf('Frame %d', frameId));
        xlabel('Azimuth');
        ylabel('Elevation');
        colorbar;
        drawnow;
        %pause(0.2);

    end

    nextFrameId = firstFrameId + length(clusters);
end