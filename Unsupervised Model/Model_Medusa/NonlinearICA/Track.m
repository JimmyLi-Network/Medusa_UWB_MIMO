
classdef Track < handle
    properties (SetAccess = private)
        track = {};
        frame = [];
        trackId = -1;

        trackSize = 0;
    end

    methods

        function obj = Track()
            obj.track = {};
            obj.frame = [];
            obj.trackId = Track.getUniqueTrackId();
        end

        function add(obj, frameId, points, pointData, group)
            t = TrackPoint;
            t.points = points; % Array of size (2, num_points);
            t.data = pointData;
            t.group = group;
            t.frame = frameId;
            t.centroid = mean(points, 2);
            assert(all(size(t.centroid) == [2 1]));

            obj.track{end+1} = t;
            obj.frame = [obj.frame frameId];
        end

        function fr = lastFrame(obj)
            if length(obj.track) > 0
                fr = obj.frame(end);
            else
                fr = NaN;
            end
        end

        function [pos] = lastPosition(obj)
            if length(obj.track) > 0
                pos = obj.track{end}.centroid;
            else
                pos = [NaN; NaN];
            end
        end

        function fr = frames(obj)
            fr = obj.frame;
        end

        function [points, data] = getPointsAtFrame(obj, frameId)
            if ~ismember(frameId, obj.frame)
                points = [];
                data = [];
                return;
            end

            idx = find(obj.frame == frameId);
            points = obj.track{idx}.points;
            data = obj.track{idx}.data;
        end

        function [trackObject, frameId] = getTrackPoint(obj, index)
            assert(index <= length(obj.track));
            trackObject = obj.track{index};
            frameId = obj.frame(index);
        end


        function sz = get.trackSize(obj)
            sz = length(obj.track);
        end

        function b = empty(obj)
            b = (length(obj.track) == 0);
        end
        
    end

    methods (Static, Access=private)
        function uniqueId = getUniqueTrackId()
            persistent globalTrackId;
            if isempty(globalTrackId)
                globalTrackId = 0;
            end

            uniqueId = globalTrackId;
            globalTrackId = globalTrackId + 1;
        end
    end
end





        



