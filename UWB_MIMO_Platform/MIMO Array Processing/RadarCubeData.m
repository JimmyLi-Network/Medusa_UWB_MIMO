%% Load Data
load('raw_radar_data.mat','raw_data_cube');
raw_radar_data = raw_data_cube;  % Assume dimensions: [M, totalFrames, numRx, numTx]
[M, totalFrames, numRx, numTx] = size(raw_radar_data);  % e.g., M=186, numTx=16

%% Parameters
windowLen = 100;       % Number of frames per block
fftLen = 128;
xaxis = 1:fftLen;
yaxis = -pi/2 : pi/128 : pi/2;
tx_map = [1, 1, 2, 3, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];  % Mapping for array frames

%% Loop over blocks of 'windowLen' frames
for n = 1:windowLen:(3001 - windowLen + 1)
    % Extract a block of frames: size [M, windowLen, numRx, numTx]
    radar_block = raw_radar_data(:, n:n+windowLen-1, :, :);
    
    % Process each frame in the current block
    for i = 1:windowLen
        % Preallocate cell array for the 16 array frames
        array_frames = cell(1, 16);
        for idx = 1:16
            array_frames{idx} = [];
        end
        
        % Accumulate data from each receiver for each desired TX channel
        for rx = 1:numRx
            for idx = 1:16
                tx = tx_map(idx);
                array_frames{idx} = [array_frames{idx}, squeeze(radar_block(:, i, rx, tx))];
            end
        end
        
        % Process each range bin to build radar_cube
        radar_cube = zeros(fftLen, fftLen, M);
        for bin = 1:M
            % Build dataplane: each row from a different array frame
            dataplane = cellfun(@(A) A(bin,:), array_frames, 'UniformOutput', false);
            dataplane = cell2mat(dataplane');  % Expected size: 16x16
            
            % Compute 2D FFT along rows and columns
            fft2d_frame = fft(dataplane, fftLen, 1);
            fft2d_frame_2 = fft(fft2d_frame, fftLen, 2);
            
            % Circularly shift to center the frequency components
            radar_cube(:,:,bin) = circshift(fft2d_frame_2, [64, 64]);
        end
        
        % Optionally, store the processed radar_cube for this frame
        radar_cube_array(:,:,:,i) = radar_cube;
    end
    % (Optional visualization or further processing can be added here.)
end
