function [N, TPs, TPTotal, FPs, FPTotal] = hogSvmModel(normSize, threshold, doOpenTest, trainRatio)
% Run HOG+SVM model once

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
[trainImages, trainLabels, testImages, testLabels] = loadSource(sourceDir, normSize, trainRatio);



%% Extract features £¨HOG)
trainFeatures = featureExtractor(trainImages);
testFeatures = featureExtractor(testImages);


%% Train SVM classifier
% reference: https://ww2.mathworks.cn/help/stats/fitcecoc.html
% train an error-correcting output codes (ECOC) multiclass model using
% support vector machine (SVM) binary learners.

svm = fitcecoc(trainFeatures, trainLabels, 'FitPosterior', 1);


%% Predict labels
[predictedLabels, negLoss, ~, posteriors] = predict(svm, testFeatures);


%% Reject negative
if doOpenTest
    sorted = sort(posteriors, 2, 'descend');
    outdraws = sorted(:,1) - sorted(:,2);
    predictedLabels(outdraws <= threshold) = '13';
end


%% Calculate metrics
N = validate(testLabels, predictedLabels);

if doOpenTest
    [TPs, TPTotal, FPs, FPTotal] = openTestPositiveMetrics(N);
else
    [TPs, TPTotal, FPs, FPTotal] = positiveMetrics(N);
end

end

