set(gcf,'position',[374.7142857142857,350.1428571428571,807.4285714285712,305.7142857142858])
x = 1:1:600;
plot(x,sig_sampled,'LineWidth',2,'Color','#0072BD');
hold on
plot(x,respiration_sig,':','LineWidth',2,'Color','#D95319');
xlabel("Time(s)");ylabel("Magnitude");ylim([3 12]);
title('The human subject is behind the wall');
box on
grid on
ax = gca;
ax.FontSize = 14;
legend('Recovered waveform','Ground truth signal')