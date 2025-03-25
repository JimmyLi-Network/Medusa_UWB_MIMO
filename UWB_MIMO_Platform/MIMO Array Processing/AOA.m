data = load('radardata_1x16.mat');
frame_processed = data.radar_frames;  % size: [186, 3001, 16]
[M, numFrames, numSensors] = size(frame_processed);  % M = 186, numFrames = 3001, numSensors = 16

%% Precompute Window Function
w = chebwin(M) .* bohmanwin(M);  % elementwise product, size: [186,1]

%% Apply Window to Each Sensor's Data
% Multiply each frame for each sensor by the window
% Using bsxfun to apply w along the first dimension
radar_frame_processed = bsxfun(@times, frame_processed, w);

%% Rearrange Data into Frames
% Create a 3D matrix: each frame is M x numSensors (186 x 16)
dataframes = permute(radar_frame_processed, [1, 3, 2]);  % size: [186, 16, 3001]

%% Compute 2D FFT for Each Frame
fftLen = 128;
fft2d_frames = zeros(M, fftLen, numFrames);
for i = 1:numFrames
    % For each frame, apply FFT along the sensor dimension for each row
    fft2d_frames(:,:,i) = fft(dataframes(:,:,i), fftLen, 2);
end

%% Circular Shift of FFT Data and Compute Average Frame
for i = 1:numFrames
    fft2d_frames(:,:,i) = circshift(fft2d_frames(:,:,i), [0, 64]);
end
average_frame = mean(fft2d_frames, 3);

%% Visualization
figure;
for i = 1:numFrames
    imagesc(abs(fft2d_frames(:,:,i)));
    xticks([-pi/2, 0, pi/2]);
    xticklabels({'-\pi/2','0','\pi/2'});
    xlabel('Azimuth');
    ylabel('Distance');
    colorbar('Ticks',[0, 0.065], 'TickLabels',{'MIN','MAX'});
    set(gca, 'FontSize', 14);
    drawnow;
end