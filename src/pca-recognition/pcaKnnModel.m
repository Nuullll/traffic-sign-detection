function [N, TPs, TPTotal, FPs, FPTotal] = pcaKnnModel(normSize, energyThreshold, K, distanceThreshold, doOpenTest)
% Run pca + knn model once

%% Load data
dataDir = '../../../dataset/data';

[trainImages, trainLabels, testImages, testLabels] = loadData(dataDir, normSize);
if doOpenTest
    % open test
    [negativeImages, negativeLabels] = loadNegative(dataDir, normSize);
    nNegative = length(negativeLabels);
    testImages(:,:,:,end+1:end+nNegative) = negativeImages;
    testLabels(end+1:end+nNegative) = negativeLabels;
end

% load source %%%%%%%%%%
sourceDir = '../../../dataset/source';
trainRatio = 0.8;
[trainImages, trainLabels, testImages, testLabels] = loadSource(sourceDir, normSize, trainRatio);


%% Preprocessing
trainFeatures = preprocess(trainImages);
testFeatures = preprocess(testImages);


%% PCA
[coeff, trainScore, trainMu] = pcaEnergy(trainFeatures, energyThreshold);
% apply pca to test data
testScore = (testFeatures - trainMu) * coeff;


%% KNN
predictedLabels = knnTest(trainScore, trainLabels, testScore, K, distanceThreshold);


%% Result analysis
N = validate(testLabels, predictedLabels);

if doOpenTest
    [TPs, TPTotal, FPs, FPTotal] = openTestPositiveMetrics(N);
else
    [TPs, TPTotal, FPs, FPTotal] = positiveMetrics(N);
end


end

