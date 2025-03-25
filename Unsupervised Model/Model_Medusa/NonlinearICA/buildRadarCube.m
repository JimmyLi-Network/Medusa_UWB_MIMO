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
radarData = radarData .* blackman(numRangeBins);
assert(numRx == numTx);

numElevationBins = numRx;
numAzimuthBins = numRx;
radarCube = zeros(numElevationBins, numAzimuthBins, numRangeBins, numFrames);

% disp('Applying window filter');
% win0 = blackman(numRangeBins);
% win1 = bohmanwin(numRangeBins);
% for f = 1:numFrames
%     for t = 1:numTx
%         for r = 1:numRx
%             radarData(:,f,r,t) = squeeze(radarData(:,f,r,t)) .* win0 .* win1;
%         end
%     end
% end

disp('FFT');
for frameId = 1:numFrames
    for rangeId = 1:numRangeBins
        v = squeeze(radarData(rangeId, frameId, :, :));
        d = fft2(v);
        radarCube(:,:,rangeId, frameId) = rot90(d, 1);

        %dd = circshift(d, [numRx/2 numRx/2]);
        %radarCube(:,:,rangeId, frameId) = rot90(dd, 1);
        %radarCube(:,:,rangeId, frameId) = fftshift(fftshift(fft2(v), 1), 2);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Filter the data using mobility
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We need to filter out static reflectors in the environment. There are 
% many reflectors that can have higher energy returns than that from the human
% target. Isolate human target by finding bins that change with motion.
%
% We ignore the first few bins, because those are polluted with self-interference
%ignoredBins = 1;
%radarCube = radarCube(:,:,ignoredBins:end,:);

radarCubeAbs = abs(radarCube);

%
% radarCubeDiff has a size of (numElevationBins, numAzimuthBins, numRangeBins, numFrames);
radarCubeDiff = radarCube(:,:,:,2:end) - radarCube(:,:,:,1:end-1);
% radarCubeDiff = radarCube(:,:,:,2:end) - radarCube(:,:,:,1:end-1);
% assert(all(size(radarCubeDiff) == [numElevationBins numAzimuthBins numRangeBins, numFrames-1]));

disp('Plotting');
% Plot the range
midpoint = floor(numRangeBins/2);
radarCubeDiffPow = abs(squeeze(sum(radarCubeDiff, 3)));
assert(all(size(radarCubeDiffPow) == [numElevationBins, numAzimuthBins, numFrames-1]));

% Find the tracking 
figure(1);
imagesc(squeeze(radarCubeAbs(9,:,:)));
xlabel('Frames');
ylabel('Range Bins');
colorbar;

cmax = max(radarCubeDiffPow, [], 'all');
cmin = min(radarCubeDiffPow, [], 'all');
clims = [cmin cmax];

cfar = phased.CFARDetector2D('GuardBandSize', 1, 'TrainingBandSize', 2, ...
        'ThresholdFactor', 'Auto', ...
        'ProbabilityFalseAlarm', 0.25)
gap = cfar.GuardBandSize + cfar.TrainingBandSize+1;
[rowInd, colInd] = meshgrid(gap:(numElevationBins - gap),...
         gap:(numAzimuthBins - gap));
cutInd = [rowInd(:) colInd(:)]';

motionTracker = MotionTracker;

figure(2);
% % Resize the figure
% currPos = get(gcf, 'Position');
% set(gcf, 'Position', [currPos(1) currPos(2) 2*currPos(3) currPos(4)]);

target = struct([]);

% Filter window size
H = ones(3,3,20,1);
for f = 1:numFrames-1
    y = squeeze(radarCubeDiffPow(:,:,f));
    target(end+1).frame = f;
   
    % subplot(121);
    imagesc(y, clims);
    det = cfar(y, cutInd);
    coord = cutInd(:, det);
    if length(coord) > 0
        idx = dbscan(coord', 2, 2);

        framePoints = coord;
        framePointData = radarCubeDiff(:,:,:,f);
        %motionTracker.addPoints(f, framePoints, framePointData, idx);

        % Get the result of tracking
        %[points, pointData, trackIds] = motionTracker.getPoints(f);
        %disp(sprintf('%s', size(points)));

        % subplot(121);
        hold on;
        % for pidx = 1:length(points)
        %     p = points{pidx};
        %     gscatter(p(2,:), p(1,:), ones(1, size(p, 2)) * trackIds(pidx));
        % end
        gscatter(coord(2,:), coord(1,:), idx);
        %gscatter(points(2,:), points(1,:), trackIds);
        pointData = containers.Map('KeyType', 'int64', 'Valuetype', 'any');
        pointDataDiff = containers.Map('KeyType', 'int64', 'Valuetype', 'any');

        %coordData = containers.Map('KeyType', 'int64', 'Valuetype', 'any');
        coordData = struct([]);

        for i = unique(idx)'
            c = coord(:, idx == i);
            maxCoord = max(c, [], 2);
            minCoord = min(c, [], 2);
            A = abs(radarCubeDiff(minCoord(1):maxCoord(1), minCoord(2):maxCoord(2), :, f));
            B = imfilter(A, H);

            % find the max element
            [M, I] = max(B, [], 'all');
            [xi, yi, zi, wi] = ind2sub(size(B), I);

            x = xi + minCoord(1) - ((size(H,1)-1)/2) - 1;
            y = yi + minCoord(2) - ((size(H,2)-1)/2) - 1;
            z = max(zi - ((size(H,3)-1)/2)-1, 1);

            % Need to subtract by one because the height and width in "rectangle"
            % does not include the starting coordinate.
            w = size(H, 1); 
            h = size(H, 2);
            l = size(H, 3);
            rectangle('Position', [y x h-1 w-1]);
            text(y,x, sprintf('x=%d, y=%d, z=%d, id=%d\nxrange=(%d,%d)\nyrange=(%d,%d)', x, y, z, i, ...
                minCoord(1), maxCoord(1), minCoord(2), maxCoord(2)));

            coordData(end+1).index = i;
            coordData(end).x = [x, x+w-1];
            coordData(end).y = [y, y+h-1];
            coordData(end).z = [z, z+l-1];

            % if i ~= -1
            %     zrange(end+1, :) = [z,z+l-1];
            % end

            d = radarCube(x:(x+w-1), y:(y+h-1), z:(z+l-1), f);
            pointData(i) = d;
            
            dd = radarCubeDiff(x:(x+w-1), y:(y+h-1), z:(z+l-1), f);
            pointDataDiff(i) = dd;
            % datablock{pidx} = d;
        end
        hold off;

        motionTracker.addPoints(f, coord, pointData, idx);
        target(end).coord = coordData;
        target(end).points = pointData;
        target(end).diffPoints = pointDataDiff;
    else
        disp(sprintf('Frame %d blank', f));
        target(end).coord = struct([]);
        target(end).points = [];
        target(end).diffPoints = [];
    end

    title(sprintf('Frame %d', f));
    xlabel('Azimuth');
    ylabel('Elevation');
    colorbar;
    drawnow;
    % pause(0.2);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save a list of radar targets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

zrange = zeros([0,2]);
zindex = [];
signal = [];
for tidx = 1:length(target)
    t = target(tidx);
    s = [];
    z = zeros([0,2]);
    for cidx = 1:length(t.coord)
        c = t.coord(cidx);
        if c.index < 0
            continue;
        end
        v = t.diffPoints(t.coord(cidx).index);
        s(end+1) = sum(abs(v), 'all');
        z(end+1,:) = t.coord(cidx).z;
    end

    if length(s) > 0
        [M, I] = max(s);
        zrange(end+1,:) = z(I,:);
        signal(end+1) = M;
    end
end

targetFile = fullfile(rawBaseDir, 'targets.mat');
if exist(targetFile, 'file')
    delete(targetFile);
end
save(targetFile, 'target', 'zrange', 'signal', '-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the set of bins that provide the greatest amount of energy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

zunique = unique(zrange, 'rows');
channelPower = zeros(1, length(zunique));
for i = 1:length(zunique)
    zi = zunique(i, 1); % + ignoredBins;
    zj = zunique(i, 2); % + ignoredBins;
    s = radarData(zi:zj,:,:,:);
    channelPower(i) = sum(abs(s), 'all');;;;
end

[M, I] = max(channelPower);
zi = zunique(I, 1); % + ignoredBins;
zj = zunique(I, 2); % + ignoredBins;
channels = radarData(zi:zj, :,:,:);
channelFile = fullfile(rawBaseDir, 'channels.h5');
if exist(channelFile, 'file')
    delete(channelFile)
end

indexRange = [zi, zj];

h5create(channelFile, '/numTx', size(numTx));
h5write(channelFile, '/numTx', numTx);
h5create(channelFile, '/numRx', size(numRx));
h5write(channelFile, '/numRx', numRx);
h5create(channelFile, '/indexRange', size(indexRange));
h5write(channelFile, '/indexRange', indexRange);
h5create(channelFile, '/numFrames', size(numFrames));
h5write(channelFile, '/numFrames', numFrames);
h5create(channelFile, '/data/I', size(channels));
h5write(channelFile, '/data/I', real(channels));
h5create(channelFile, '/data/Q', size(channels));
h5write(channelFile, '/data/Q', imag(channels));

% 
% zcoord = unique(zrange(:,1));
% signal = [];
% for idx = 1:length(zcoord)
%     zi = zcoord(idx);
%     zj = zi + size(H, 3);
%     s = radarData(zi:zj,:,:,:);
%     signal(idx) = sum(abs(s), 'all');
% end
% 
% [M, I] = max(signal);
% zi = zcoord(I);
% zj = zi + size(H,3);
% channels = squeeze(radarData(zi:zj, :, :,:));
% channelFile = fullfile(rawBaseDir, 'channels.h5');
% if exist(channelFile, 'file')
%     delete(channelFile)
% end
% 
% h5create(channelFile, '/numTx', size(numTx));
% h5write(channelFile, '/numTx', numTx);
% h5create(channelFile, '/numRx', size(numRx));
% h5write(channelFile, '/numRx', numRx);
% h5create(channelFile, '/numFrames', size(numFrames));
% h5write(channelFile, '/numFrames', numFrames);
% h5create(channelFile, '/data/I', size(channels));
% h5write(channelFile, '/data/I', real(channels));
% h5create(channelFile, '/data/Q', size(channels));
% h5write(channelFile, '/data/Q', imag(channels));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate the radar cube
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get the distribution of the track length
trackSize = zeros(1, length(motionTracker.tracks));
for i = 1:length(motionTracker.tracks)
    trackSize(i) = motionTracker.tracks{i}.trackSize;
end

[M,I] = sort(trackSize, 'descend');

numTracks = 10;
datablocks = {};
trackId = zeros(1, numTracks);
for i = 1:numTracks
    track = motionTracker.tracks{I(i)};
    trackId(i) = motionTracker.tracks{I(i)}.trackId;
    B = [];
    for p = 1:track.trackSize
        [t,f] = track.getTrackPoint(p);
        d = t.data;
        %s = reshape(d, [size(d,1)*size(d,2), size(d,3)]);
        %s = reshape(s, [1 size(s)]);
        s = reshape(d, [1 numel(d)]);
        B = cat(1, B, s);
    end
    datablocks{end+1} = B;
end

trackFile = fullfile(rawBaseDir, trackOutputFile);
if exist(trackFile, 'file')
    delete(trackFile);
end

h5create(trackFile, '/trackId', size(trackId));
h5write(trackFile, '/trackId', trackId);
for i = 1:numTracks
    ds = sprintf('/tracks/track_%d', trackId(i));
    dsReal = sprintf('%s/I', ds);
    dsImag = sprintf('%s/Q', ds);
    h5create(trackFile, dsReal, size(datablocks{i}));
    h5write(trackFile, dsReal, real(datablocks{i}));
    h5create(trackFile, dsImag, size(datablocks{i}));
    h5write(trackFile, dsImag, imag(datablocks{i}));
end
