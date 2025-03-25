figure(1);
subplot(2,1,1);
plot(sig_sampled);
xlabel("Time(s)");ylabel("Recovered signal magnitude");title(strcat("Recovered respiration signal of jogging behind the wall (TX-RX: 6-7) CosSim:",num2str(cosSim)));

%subplot(3,1,2);
%plot(sig_sampled2);
%xlabel("Time(s)");ylabel("Recovered signal magnitude");title(strcat("Recovered respiration signal of jogging behind the wall (TX-RX: All) CosSim:",num2str(cosSim2)));

subplot(2,1,2);
respiration_filtered = sgolayfilt(respiration_sig2,9,13);
plot(respiration_filtered);
xlabel("Time(s)");ylabel("Force(N)");title("Ground truth signal");