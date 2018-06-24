function [outputArg1,outputArg2] = hogSvmModel(normSize,inputArg2)
% Run HOG+SVM model once

%% Load data
dataDir = '../../../dataset/data';

[trainImages, trainLabels, testImages, testLabels] = loadData(dataDir, normSize);


%% Extract features
trainFeatures = featureExtractor(trainImages);
testFeatures = featureExtractor(testImages);

end

