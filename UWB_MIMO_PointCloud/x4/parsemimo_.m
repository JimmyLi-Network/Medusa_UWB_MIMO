% close all;
bin_offerset = 10;
n_tx = 4;
n_rx = 4;
RECTANGULAR_ANTENNA = 1;

win = 32;


h = figure(5);
set(h,'position',[50 350 1800 500]);

%frame*rx*bin
raw_data = load("data_5.mat");
frame_org = raw_data.frame_org;

%tx * rx * frame * bin
frame = permute(frame_org,[2,3,1,4]);
%frame_array = permute(frame_org,[3,4,2,1]);


%remove background
frame_rm_dc = zeros(size(frame));
for j = 1:n_tx
    for k = 1:n_rx
        frame_temp = squeeze(frame(j,k,:,:));
        frame_temp = phase_correction(frame_temp);
        frame_temp(:,1:bin_offerset) = 0;
        frame_rm_dc(j,k,:,:) = frame_temp - mean(frame_temp);  
    end
end

%doppler
bin_cnt = 138;
doppler_fft_size = 128;
doppler_result = zeros(n_tx,n_rx,doppler_fft_size,bin_cnt);
for j = 1:n_tx
    for k = 1:n_rx
        for l = 1:bin_cnt
            doppler_temp = squeeze(frame_rm_dc(j,k,:,l)) .* hamming(win);
            doppler_result(j,k,:,l) = fftshift(fft(doppler_temp,doppler_fft_size));
        end
    end
end


%先CFAR 找出符合目标的点
doppler_result_temp = abs(squeeze(mean(doppler_result,[1,2])));
doppler_result_temp_sum = sum(doppler_result_temp(20:100,:),1);
[range_max,range_idx] = max(doppler_result_temp_sum);
distance = (range_idx - 10) * 0.0514;
%findpeaks
[pks,locs] = findpeaks(doppler_result_temp_sum,'SortStr','descend');

doppler_result_x = [];
doppler_result_y = [];
for j = 1:2
    range_idx = [];
    pk = pks(j);loc = locs(j);
    if pk < range_max * 0.9
        break
    end
    range_idx = [loc-4,loc-3,loc-2,loc-1,loc,loc+1,loc+2,loc+3,loc+4,loc+5];
    %range_idx = [loc-2,loc-1,loc,loc+1,loc+2];
    for k = 1:length(range_idx)
        temp_data = doppler_result_temp(:,range_idx(k));
        sort_data = sort(temp_data);
        cfar_thre = sort_data(end - 30);
        [temp_x_idx,~] = find(temp_data > cfar_thre);
        temp_y_idx = ones(size(temp_x_idx)) * range_idx(k);
        doppler_result_x = [doppler_result_x;temp_x_idx];
        doppler_result_y = [doppler_result_y;temp_y_idx];
    end
end


%calculate azimuth and elevation
azimuth_fft_size = 180;
elevation_fft_size = 180;
azimuth_angle_list = [];
elevation_angle_list = [];
range_list = [];
speed_list = [];

%azimuth and elevation
for k = 1:length(doppler_result_x)
    doppler_freq_idx = doppler_result_x(k);
    doppler_range_idx = doppler_result_y(k);
    if RECTANGULAR_ANTENNA == 1 %矩形天线
        if n_rx >= 3 
            data_to_fft_1 = squeeze(doppler_result(1,1:2,doppler_freq_idx,doppler_range_idx));
            data_to_fft_2 = squeeze(doppler_result(2,1:2,doppler_freq_idx,doppler_range_idx));
            data_to_fft = cat(2,data_to_fft_1,data_to_fft_2);

            angle_map = abs(fftshift(fft(data_to_fft,azimuth_fft_size)));
            [~,azimuth_idx] = find(angle_map == max(angle_map,[],[1,2]));

            data_to_fft = squeeze(doppler_result(1,1:2:3,doppler_freq_idx,doppler_range_idx));
            angle_map = abs(fftshift(fft(data_to_fft,elevation_fft_size)));
            [~,elevation_idx] = find(angle_map == max(angle_map,[],[1,2]));

            %azimuth
            azimuth_angle = get_anle(azimuth_idx,azimuth_fft_size);
            %elevation
            elevation_angle = get_anle(elevation_idx,elevation_fft_size);
        elseif n_rx >=2
            data_to_fft = squeeze(doppler_result(1,1:2,doppler_freq_idx,doppler_range_idx));
            angle_map = abs(fftshift(fft(data_to_fft,azimuth_fft_size)));
            [~,azimuth_idx] = find(angle_map == max(angle_map,[],[1,2]));

            %azimuth
            azimuth_angle = get_anle(azimuth_idx,azimuth_fft_size);
            elevation_angle = 0;
        else
            azimuth_angle =0;
            eelevation_angle = 0;
        end  
    else
        if n_rx >= 2 && n_tx >= 2
            data_to_fft = squeeze(doppler_result(:,:,doppler_freq_idx,doppler_range_idx));
            angle_map = abs(fftshift(fft2(data_to_fft,elevation_fft_size,azimuth_fft_size)));
            [elevation_idx,azimuth_idx] = find(angle_map == max(angle_map,[],[1,2]));

            %azimuth
            azimuth_angle = get_anle(azimuth_idx,azimuth_fft_size);
            %elevation
            elevation_angle = get_anle(elevation_idx,elevation_fft_size);
         elseif n_rx >=2
            data_to_fft = squeeze(doppler_result(:,:,doppler_freq_idx,doppler_range_idx));
            angle_map = abs(fftshift(fft(data_to_fft,azimuth_fft_size)));
            [elevation_idx,azimuth_idx] = find(angle_map == max(angle_map,[],[1,2]));

            %azimuth
            azimuth_angle = get_anle(azimuth_idx,azimuth_fft_size);
            elevation_angle = 0;
        elseif n_tx >=2
            data_to_fft = squeeze(doppler_result(:,:,doppler_freq_idx,doppler_range_idx));
            angle_map = abs(fftshift(fft(data_to_fft,elevation_fft_size)));
            [elevation_idx,azimuth_idx] = find(angle_map == max(angle_map,[],[1,2]));

            %elevation
            azimuth_angle = 0;
            elevation_angle = get_anle(elevation_idx,elevation_fft_size);
        else
            azimuth_angle = 0;
            elevation_angle = 0;
        end
    end
    azimuth_angle_list = [azimuth_angle_list;azimuth_angle];
    elevation_angle_list = [elevation_angle_list;elevation_angle];
    %range
    range_list = [range_list;(doppler_range_idx - 10) * 0.0514];
    %speed
    speed_list = [speed_list;0];
end

%theta1 =  real(asin(ll*(angle_mean1)/2/pi/d)*180/pi);

fprintf("position:%d,azimuth:%d,elevation:%d\n",doppler_result_y(1),azimuth_angle,elevation_angle);

plot_x = range_list .* sin(deg2rad(azimuth_angle_list));
plot_y = range_list .* cos(deg2rad(azimuth_angle_list));

plot_z = range_list .* sin(deg2rad(elevation_angle_list));


degs = -90:10:90;
dists = 0:1:5;

drawnow;
subplot(131)
mesh(abs(squeeze(frame_rm_dc(1,1,:,:))));
title("time-range")
ylabel("time")
xlabel("range(bin)")

subplot(132)
mesh(abs(squeeze(mean(doppler_result,[1,2]))))
title("doppler-fft")
ylabel("fft index")
xlabel("range(bin)")

subplot(133)
%plot(plot_x,plot_y,'LineStyle','none','Marker','o','MarkerSize',10,'MarkerFace','y','MarkerEdge',[1,0,0],'LineWidth',2)

plot3(plot_x,plot_y,plot_z,'LineStyle','none','Marker','o','MarkerSize',10,'MarkerFace','y','MarkerEdge',[1,0,0],'LineWidth',2)
hold on
plot3(plot_x,plot_y,ones(size(plot_z)) * (min(plot_z) - 0.1) ,'LineStyle','none','Marker','o')
grid on
xlim([-3,3])
ylim([0,5])

title("orientation diagram")
ylabel("Y(m)")
xlabel("X(m)")
zlabel("Z")


function last_angle = get_anle(angle_idx,fft_size)
    fw = (angle_idx - fft_size / 2 - 1) / fft_size;
    theta = asin(fw * 2);
    last_angle = round(theta * 180 / pi);
end

% 相位校准
function org_corr = phase_correction(org)
    bin_ref = 10;   
    [FPS,BIN] = size(org);   
    bin_angle = mean(angle(org(:,bin_ref)));
    org_corr = zeros(FPS,BIN);
    for i = 1:FPS
        phase_correction = bin_angle - angle(org(i,bin_ref));
        org_corr(i,:) = org(i,:) * exp(1i*phase_correction);
    end
end

