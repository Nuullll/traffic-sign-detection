function [N, TPs, TPTotal, FPs, FPTotal] = pcaKnnModel(normSize, energyThreshold, K)
% Run pca + knn model once

%% Load data
dataDir = '../../../dataset/data';

[trainImages, trainLabels, testImages, testLabels] = loadData(dataDir, normSize);


%% Preprocessing
trainFeatures = preprocess(trainImages);
testFeatures = preprocess(testImages);


%% PCA
[coeff, trainScore, trainMu] = pcaEnergy(trainFeatures, energyThreshold);
% apply pca to test data
testScore = (testFeatures - trainMu) * coeff;


%% KNN
predictedLabels = knnTest(trainScore, trainLabels, testScore, K);


%% Result analysis
N = validate(testLabels, predictedLabels);

[TPs, TPTotal, FPs, FPTotal] = positiveMetrics(N);


end

