function labels = knnTest(trainData, trainLabels, testData, K, threshold)
% Test using KNN
% Input:    trainData       
% Input:    trainLabels
% Input:    testData
% Input:    K           number of nearest neighbours
% Input:    threshold   distance threshold to reject negative samples
% Output:   labels      predicted labels

% normalize
vars = var(trainData);
trainData = trainData ./ vars;
testData = testData ./ vars;

nTest = size(testData, 1);

labels = zeros(nTest, 1);

% knn kernel
for i = 1:nTest
    sample = testData(i,:);
    distances = sqrt(sum((trainData - sample).^2, 2));
    [~,I] = sort(distances);
    sorted = distances(I(1:K));
    sorted = sorted(sorted <= threshold);
    K_ = length(sorted);
    if K_ == 0
        labels(i) = 13;
    else
        labels(i) = mode(trainLabels(I(1:K_)));
    end
end

labels = categorical(labels);

end

