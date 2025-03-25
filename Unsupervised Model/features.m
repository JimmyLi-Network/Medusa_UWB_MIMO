load('performance.mat')

figure(1);
subplot(1,2,1);
models = [2,4,6];
y  = [numberOfPulses_1,numberOfPulses_2,numberOfPulses_3];
err = [err_1,err_2,err_3];

errorbar(models,y,err,'bs');
ylim([0 80]);xlim([0 10]);xticks([2,4,6]);xticklabels({'attention', 'main', 'new model 3'});
title('Respiration estimation error');
ylabel('Respiration estimation error');
str1 = {'   GT:'}; % add a label
str2 = num2str(respiration_count);
str3 = {' ,  VMD:'};
str4 = num2str(numberOfPulses_1);
str = strcat(str1,str2,str3,str4);
text(2,numberOfPulses_1,str);

str3 = {' ,  VMD:'};
str4 = num2str(numberOfPulses_2);
str = strcat(str3,str4);
text(4,numberOfPulses_2,str);

str3 = {' ,  VMD:'};
str4 = num2str(numberOfPulses_3);
str = strcat(str3,str4);
text(6,numberOfPulses_3,str);



%cosine similarity
%sig_cutted = sig(1:3001/length(respiration_sig):3001); % keep the two arrays with the same length
cosSim_1 = dot(sig_formed_1,respiration_sig)/(norm(sig_formed_1)*norm(respiration_sig)); % calculate cosine similarity using the method in paper
cosSim_2 = dot(sig_formed_2,respiration_sig)/(norm(sig_formed_2)*norm(respiration_sig));
cosSim_3 = dot(sig_formed_3,respiration_sig)/(norm(sig_formed_3)*norm(respiration_sig));

subplot(1,2,2);
err_rate_1 = abs(numberOfPulses_1 - respiration_count) / respiration_count;
err_rate_2 = abs(numberOfPulses_2 - respiration_count) / respiration_count;
err_rate_3 = abs(numberOfPulses_3 - respiration_count) / respiration_count;

cosSim = [cosSim_1,cosSim_2,cosSim_3];
err_rate = [err_rate_1 err_rate_2, err_rate_3];
errorbar(models,cosSim,err_rate,'ro');
title('Cosine similarity and error rate');
ylim([0 1]);xlim([0 10]);yticks([0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]);yticklabels({'0%','10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'});
xticks([2,4,6]);xticklabels({'attention', 'main', 'new model 3'});
ylabel('Cosine similarity with relative error');
str1 = {'  Error rate:'}; % add a label
str2 = sim2str(vpa(err_rate_1,3));
str3 = {' , CosSim: '};
str4 = num2str(cosSim_1);
str = strcat(str1,str2,str3,str4);
text(2,cosSim_1,str);

str1 = {'  Error rate:'}; % add a label
str2 = sim2str(vpa(err_rate_2,3));
str3 = {' , CosSim: '};
str4 = num2str(cosSim_2);
str = strcat(str1,str2,str3,str4);
text(4,cosSim_2,str);

str1 = {'  Error rate:'}; % add a label
str2 = sim2str(vpa(err_rate_3,3));
str3 = {' , CosSim: '};
str4 = num2str(cosSim_3);
str = strcat(str1,str2,str3,str4);
text(6,cosSim_3,str);

