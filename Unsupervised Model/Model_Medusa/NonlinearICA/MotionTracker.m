
classdef MotionTracker < handle
    properties (SetAccess = private)
        tracks = {};
        frameIds = []; % List of frame ids that are cached in the tracker            
    end
    
    properties (Constant)
        RangeBinSize = 0.05144; % 5cm per range bin
    end

    methods

        function obj = MotionTracker()
            obj.tracks = {};
        end

        function addPoints(obj, frameId, points, pointData, indices)
            assert(size(points,1) == 2); 
            %numPoints = size(points, 2); % Length of stream
            %assert(all(size(pointData, [1 2]) == [numPoints numPoints]));

            trackIds = obj.getTrackIdsFromFrameRange(frameId-1, frameId-1);
            centroid = obj.getTrackCentroid(trackIds);

            trackMap = -1 * ones(1, length(trackIds));
            trackVal = inf * ones(1,length(trackIds));
            trackPoints = {}; 
            trackData = {};
            obj.frameIds(end+1) = frameId;

            groups = unique(indices);
            for i = 1:length(groups)
                g = groups(i);
                if g == -1
                    continue;
                end

                p = points(:, indices == g);
                c = mean(p, 2);
                assert(all(size(c) == [2 1]));

                % The radar cube array has the dimensions (elevation, azimuth, range)
                %d = pointData(p(1,:), p(2,:), :);

                rangeData = [];
                if isa(pointData, 'containers.Map')
                    rangeData = pointData(g);
                else
                    rangeData = obj.filterPoints(p, pointData);
                end

                if length(trackIds) == 0
                    newTrack = Track;
                    newTrack.add(frameId, p, rangeData, g);
                    obj.tracks{end+1} = newTrack;
                else
                    assert(size(centroid, 1) == 2);
                    centroidDiff = centroid - c;
                    assert(all(size(centroidDiff) == [2, length(trackIds)]));
                    centroidDist = sqrt(sum(centroidDiff.^2, 1));
                    
                    [minVal, minIdx] = min(centroidDist);
                    if trackVal(minIdx) > minVal
                        % trackMap maps a track index to a group in the current frame
                        trackMap(minIdx) = i;
                        trackVal(minIdx) = minVal;
                        trackPoints{minIdx} = p;
                        trackData{minIdx} = rangeData;
                    end
                end
            end

            for i = 1:length(trackMap)
                groupId = trackMap(i);
                if groupId == -1
                    continue;
                end
                obj.tracks{trackIds(i)}.add(frameId, trackPoints{i}, trackData{i}, groupId);
            end

            % Find out which groups in the current frame have not been mapped 
            % by any existing track. 
            unmatchedGroupIds = setdiff(1:length(groups), trackMap);

            % Create a new track for each of these unmatched groups
            for i = 1:length(unmatchedGroupIds)
                g = groups(unmatchedGroupIds(i));
                p = points(:, indices == g);
                %d = pointData(p(1,:), p(2,:), :);
                %rangeData = obj.filterPoints(p, pointData);
                if isa(pointData, 'containers.Map')
                    rangeData = pointData(g);
                else
                    rangeData = obj.filterPoints(p, pointData);
                end

                newTrack = Track;
                newTrack.add(frameId, p, rangeData, g);
                obj.tracks{end+1} = newTrack;
            end
        end

        % Retrieves all tracks that are active/alive at frameId
        function [points, data, trackIds] = getPoints(obj, frameId)
            points = {};
            data = {};
            trackIds = [];
            for i = 1:length(obj.tracks)
                t = obj.tracks{i};
                [p, d] = t.getPointsAtFrame(frameId);
                if length(p) > 0
                    points{end+1} = p;
                    data{end+1} = d;
                    trackIds = [trackIds t.trackId];
                end
            end
        end

        % Retrieves all tracks that are active/alive at frameId
        function tracks = getTracks(obj, frameId)
            tracks = []; 
            for i = 1:length(obj.tracks)
                t = obj.tracks{i};
                [p, d] = t.getPointsAtFrame(frameId);
                if length(p) > 0
                    tracks(end+1) = t;
                end
            end
        end

    end

    methods(Access=private)
        % Retrieve the indices of tracks that fall within 
        % the selection criteria
        function trackIds = getTrackIdsFromFrameRange(obj, currFrameId, oldestFrameId)
            trackIds = [];
            for t = 1:length(obj.tracks)
                if obj.tracks{t}.empty()
                    continue
                end

                f = obj.tracks{t}.lastFrame();

                if (oldestFrameId <= f) & (f <= currFrameId)
                    trackIds = [trackIds t];
                end
            end
        end

        function data = filterPoints(obj, points, pointData)

            numPoints = size(points, 2); % Length of stream
            d = zeros(numPoints, size(pointData,3));
            for k = 1:numPoints
                d(k,:) = squeeze(pointData(points(1,k), points(2,k), :));
            end

            data = d;
        end

        function rangeData = getRangeData(obj, points, pointData)
            numPoints = size(points, 2); % Length of stream
            d = zeros(numPoints, size(pointData,3));
            for k = 1:numPoints
                d(k,:) = squeeze(pointData(points(1,k), points(2,k), :));
            end


            [maxVal, maxIdx] = max(abs(d), [], 2);
            
            rangeData = zeros(1, numPoints);
            for k = 1:numPoints
                c = maxIdx(k);
                v = d(k, c);
                range = (c*MotionTracker.RangeBinSize) + ...
                    ((angle(v)/pi) * (MotionTracker.RangeBinSize/2));
                rangeData(k) = range;
            end
        end

        %     d = data;
        %     % It takes too much memory/storage space to keep all the range information
        %     % from that point, so we just track the highest signal bin instead.
        %     [maxVal, maxIdx] = max(abs(d), [], 3);
        %     assert(all(size(d, [1 2]) == size(maxIdx)));

        %     % Compute the range information for each max point
        %     rangeData = zeros(size(maxIdx));
        %     for a = 1:size(maxIdx, 1)
        %         for b = 1:size(maxIdx, 2)
        %             c = maxIdx(a,b);
        %             v = d(a,b,c);
        %             range = (c*MotionTracker.RangeBinSize) + ...
        %                 ((angle(v)/pi) * (MotionTracker.RangeBinSize/2));
        %             rangeData(a, b) = range;
        %         end
        %     end
        % end

        function centroid = getTrackCentroid(obj, trackIds)
            centroid = {};
            for i = 1:length(trackIds)
                t = trackIds(i);
                if obj.tracks{t}.empty()
                    continue
                end
                centroid{end+1} = obj.tracks{t}.lastPosition();
            end
            centroid = cell2mat(centroid);
        end
    end
end


        
        

 
