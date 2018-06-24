function labels = knnTest(trainData, trainLabels, testData, K)
% Test using KNN
% Input:    trainData       
% Input:    trainLabels
% Input:    testData
% Input:    K           number of nearest neighbours
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
    labels(i) = mode(trainLabels(I(1:K)));
end

labels = categorical(labels);

end

