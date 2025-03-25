clear all;
close all;

% Generate new sources
numSources = 5;
sourceLength = 1024 * 1024;

source = zeros(numSources, sourceLength);

for i = 1:numSources
    source(i,:) = -abs(ar1(0.7, sourceLength));
end

save('source.mat', 'source', '-v7.3');

delete 'source.h5';
h5create('source.h5', '/source', size(source));
h5write('source.h5', '/source', source);

