% FasterRCNN uses CNN to find region proposals
% do not need a pre-trained network
% https://www.mathworks.com/help/vision/examples/object-detection-using-faster-r-cnn-deep-learning.html

%% Load GTSDB dataset (900 images with ROI labels)
gtsdbDir = fullfile(pwd, 'Origin\');
gtsdbData = loadGTSDB2Table(gtsdbDir);
nDataset = size(gtsdbData, 1);  % =1213 (some images may contain several ROI labels)

% extract by category
data = {};      % save data by categories
categoryList = gtsdbData.Properties.VariableNames(2:end);
nCategories = length(categoryList);

for i = 1:nCategories
    categoryName = categoryList{i};
    
    [roi, ind] = extractByCategory(gtsdbData, categoryName);
    
    data.(categoryName).roi = roi;
    data.(categoryName).ind = ind;
    data.(categoryName).count = length(ind);
end


%% Generate training set and test set
trainingSetRatio = 0.9;     % pick a part of whole dataset to train R-CNN

trainingIndByCategory = {};
trainingInd = [];   % all slices for training

for i = 1:nCategories
    categoryName = categoryList{i};
    
    count = data.(categoryName).count;
    randI = randsample(count, floor(trainingSetRatio * count));
    
    ind = data.(categoryName).ind(randI);
    trainingIndByCategory.(categoryName) = ind;
    trainingInd = [trainingInd; ind];
end

trainingSet = gtsdbData(trainingInd, :);

% the rest part of the whole dataset is regarded as the test set
testInd = setdiff(1:nDataset, trainingInd);
testSet = gtsdbData(testInd, :);


%% Create a CNN
% Input layer
inputLayer = imageInputLayer([32 32 3]);
% the input image size is a rough estimation of the ROI size

filterSize = [3 3];
numFilters = 32;

% Middle layers
middleLayers = [
    convolution2dLayer(filterSize, numFilters, 'Padding', 1);
    reluLayer();
    convolution2dLayer(filterSize, numFilters, 'Padding', 1);
    reluLayer();
    maxPooling2dLayer(3, 'Stride', 2);
];

% Final layers
finalLayers = [
    fullyConnectedLayer(64);
    reluLayer();
    fullyConnectedLayer(width(trainingSet));
    softmaxLayer();
    classificationLayer();
];

% Cascade all layers
layers = [
    inputLayer;
    middleLayers;
    finalLayers;
];


%% Configure training options
% trainFasterRCNNObjectDetector trains the detector in four steps. 
% The first two steps train the region proposal and detection networks.
% The final two steps combine the networks from the first two steps such
% that a single network is created for detection

% Options for step 1.
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-2, ...
    'CheckpointPath', tempdir, ...
    'Verbose', true);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-2, ...
    'CheckpointPath', tempdir, ...
    'Verbose', true);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir, ...
    'Verbose', true);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir, ...
    'Verbose', true);

options = [
    optionsStage1;
    optionsStage2;
    optionsStage3;
    optionsStage4;
];


%% Train FasterRCNN
doTraining = true;

if doTraining
    fasterRCNN = trainFasterRCNNObjectDetector(trainingSet, layers, options, ...
        'NegativeOverlapRange', [0 0.2], ...
        'PositiveOverlapRange', [0.7 1]);
    
    save('faster-rcnn-gtsdb.mat', 'fasterRCNN');
else
    load('faster-rcnn-gtsdb.mat', 'fasterRCNN');
end