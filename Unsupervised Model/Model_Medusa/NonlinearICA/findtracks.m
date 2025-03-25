
% Third processing step in the radar pipeline
% First step: showcube.m
% Second step: tracker.m
% Third step: findtracks.m

clear all;
close all;

setsource;

% Load the motion tracker from the results file saved from the previous step
motionTrackerFile = fullfile(baseDir, 'motionTrackedResult.mat');
motionTrackerData = load(motionTrackerFile);
motionTracker = motionTrackerData.motionTracker;

trackSizeDistribution(motionTracker);
[trackData, trackId] = getRange(motionTracker, 800);

matfile = fullfile(baseDir, 'trackDistances.mat');
save(matfile, 'trackData', 'trackId');

% Write a HDF5 file
h5file = fullfile(baseDir, 'trackDistances.h5');
if exist(h5file, 'file')
    delete(h5file);
end

h5create(h5file, '/trackId', size(trackId));
h5write(h5file, '/trackId', trackId);
for i = 1:length(trackData)
    ds = sprintf('/range/track_%d', i);
    h5create(h5file, ds, size(trackData{i}));
    h5write(h5file, ds, trackData{i});
end


% Find the distribution of track lengths
function trackSizeDistribution(m)
    s = [];
    for i = 1:length(m.tracks)
        s(end+1) = m.tracks{i}.trackSize;
    end
    histogram(s, 100);
    title('Track Length Distribution');
    ylabel('Count');
    xlabel('Track Length');
end

function [trackData, trackId] = getRange(m, minSize)

    % Get the tracks with size at least minSize
    trackData = {};
    trackId = [];
    for i = 1:length(m.tracks)
        if m.tracks{i}.trackSize < minSize
            continue;
        end

        t = m.tracks{i};

        minNumPoints = inf;
        for j = 1:length(t.track)
            numPoints = size(t.track{j}.points,2);
            minNumPoints = min([numPoints, minNumPoints]);
        end

        assert(minNumPoints >= 1);
        K = min([5, minNumPoints]);
        d = zeros(K, length(t.track));
        for j = 1:length(t.track)
        
            % Find the point closest to the center of mass
            assert(size(t.track{j}.points,1) == 2);
            centroid = mean(t.track{j}.points, 2);
            dist = sqrt(sum((centroid - t.track{j}.points).^2, 1));
            [B, I] = sort(dist);
            %[minDist, minIdx] = min(sqrt(sum((centroid - t.track{j}.points).^2, 2)));
            d(:, j) = t.track{j}.data(I(1:K));
        end
        trackId(end+1) = i;
        trackData{end+1} = d;
    end
    
    % Plot the trackData
    figure;
    hold on;
    for i = 1:length(trackData)
        plot(trackData{i}');
    end
    hold off;
    set(gca, 'yscale', 'log');
    grid on;
    ylabel('Distance');
    xlabel('Frame');
end
