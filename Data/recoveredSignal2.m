figure(1);
sig_normalized = normalize(sig_sampled, 'range', [0,35]);
respiration_filtered = sgolayfilt(respiration_sig2,9,13);
respiration_filtered = respiration_filtered - 11;
respiration_filtered = circshift(respiration_filtered,-11);

subplot(2,1,1);
x = 1:1:300;
plot(x,sig_normalized(1:300,:),'LineWidth',1,'Color','#0072BD');
hold on
plot(x,circshift(respiration_filtered(1:300,:),-4),':','LineWidth',1,'Color','#D95319');
xlabel("Time(s)");ylabel("Magnitude");ylim([0 40]);
title('Stand up and go out of the room');
ax = gca;
ax.FontSize = 14;
legend('Recovered waveform','Ground truth signal')

subplot(2,1,2);
x = 1:1:300;
plot(x,sig_normalized(1701:2000,:),'LineWidth',1,'Color','#0072BD');
hold on
plot(x,circshift(respiration_filtered(1701:2000,:),-2),':','LineWidth',1,'Color','#D95319');
xlabel("Time(s)");ylabel("Magnitude");ylim([0 40]);
title('Jogging on the spot');
ax = gca;
ax.FontSize = 14;
legend('Recovered waveform','Ground truth signal')

