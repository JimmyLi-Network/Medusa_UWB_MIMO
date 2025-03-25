
clear all;
close all;

fileBaseName = 'radar_raw_data';
baseDir = fullfile('data', '1tx16rx');
filename = fullfile(baseDir, strcat(fileBaseName, '.mat'));
field = 'raw_radar_data';
d = load(filename);

% Data has format (range, frame_id, rx_id)
data = getfield(d, field);

numRx = size(data, 3);
for r = 1:numRx
    fn = fullfile(baseDir, [fileBaseName '-rx' num2str(r) '.h5']);
    disp(sprintf('Writing %s', fn));
    if exist(fn, 'file')
        delete(fn);
    end
    h5create(fn, '/radar/I', size(data, [1,2]));
    h5create(fn, '/radar/Q', size(data, [1,2]));

    h5write(fn, '/radar/I', real(squeeze(data(:,:,r))));
    h5write(fn, '/radar/Q', imag(squeeze(data(:,:,r))));
end


