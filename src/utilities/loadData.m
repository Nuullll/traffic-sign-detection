function [trainImages, trainLabels, testImages, testLabels] = loadData(dataDir, normSize)
% Load train images and test images in data/
% The images will be resized to normSize

trainRecipe = fullfile(dataDir, 'train.txt');
testRecipe = fullfile(dataDir, 'test.txt');

trainTable = readtable(trainRecipe);
testTable = readtable(testRecipe);

nTrain = height(trainTable);
nTest = height(testTable);

trainImageFilenames = trainTable.Var1;
trainImageFilenames = cellfun(@(x) fullfile(dataDir, x), trainImageFilenames, ...
    'UniformOutput', false);

testImageFilenames = testTable.Var1;
testImageFilenames = cellfun(@(x) fullfile(dataDir, x), testImageFilenames, ...
    'UniformOutput', false);

% load train images
trainImages = uint8(zeros([normSize, 3, nTrain]));
for i = 1:nTrain
    filename = trainImageFilenames{i};
    image = imread(filename);
    image = imresize(image, normSize);
    trainImages(:,:,:,i) = image;
end

trainLabels = categorical(cellfun(@str2double, trainTable.Var2));

% load test images
testImages = uint8(zeros([normSize, 3, nTest]));
for i = 1:nTest
    filename = testImageFilenames{i};
    image = imread(filename);
    image = imresize(image, normSize);
    testImages(:,:,:,i) = image;
end

testLabels = categorical(cellfun(@str2double, testTable.Var2));

end

