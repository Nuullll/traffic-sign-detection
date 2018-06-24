% Fine-tune the pretrained CNN (on CIFAR-10) for traffic sign detection

%% Load GTSDB dataset (900 images with ROI labels)
gtsdbDir = fullfile(pwd, 'Origin');
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


%% Train R-CNN Traffic Sign Detector
doTraining = false;

if doTraining
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 10, ...
        'MaxEpochs', 100, ...
        'Verbose', true);
    
    rcnn = trainRCNNObjectDetector(trainingSet, cifar10Net, options, ...
        'NegativeOverlapRange', [0 0.1], 'PositiveOverlapRange', [0.8 1]);
    
    save('rcnn-gtsdb.mat', 'rcnn');
else
    load('rcnn-gtsdb.mat', 'rcnn');
end