% main script for svm-recognition

%% Define parameters
normSize = [32 32];
doOpenTest = false;
threshold = 0;

%% Run HOG+SVM model
[N, TPs, TPTotal, FPs, FPTotal] = hogSvmModel(normSize, threshold, doOpenTest);


%% Plot P matrix
P = N ./ sum(N, 2);
figure;
imagesc(P);
axis equal;
axis([0.5 12.5 0.5 12.5]);
colormap(gray);
title(sprintf('Normalized size: 32x32, TP total = %f, FP total = %f', TPTotal, FPTotal));


%% Open test
normSize = [32 32];
doOpenTest = true;

TPList = [];
rejectRateList = [];
thresholdList = 0:0.05:0.5;

for threshold = thresholdList
    [N, TPs, TPTotal, FPs, FPTotal] = hogSvmModel(normSize, threshold, doOpenTest);
    TPList(end+1) = TPTotal;
    P = N ./ sum(N, 2);
    figure;
    imagesc(P);
    axis equal;
    axis([0.5 13.5 0.5 13.5]);
    colormap(gray);
    title(sprintf('Threshold: %f', threshold));
    rejectRateList(end+1) = P(13,13);
end

%% Plot P matrix
P = N ./ sum(N, 2);
figure;
imagesc(P);
axis equal;
axis([0.5 13.5 0.5 13.5]);
colormap(gray);

