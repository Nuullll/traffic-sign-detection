% Fine-tune the pretrained CNN (on CIFAR-10) for traffic sign detection

%% Load GTSDB dataset (900 images with ROI labels)
gtsdbDir = fullfile(pwd, 'Origin\');
gtsdbData = loadGTSDB2Table(gtsdbDir);
nDataset = size(gtsdbData, 1);  % =1213 (some images may contain several ROI labels)


%% Choose part of GTSDB dataset as training set
nTrainset = 100;
trainData = gtsdbData(randsample(nDataset, nTrainset), :);


%% Train R-CNN Traffic Sign Detector
doTraining = true;

if doTraining
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-3, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 100, ...
        'MaxEpochs', 100, ...
        'Verbose', true);
    
    rcnn = trainRCNNObjectDetector(trainData, cifar10Net, options, ...
        'NegativeOverlapRange', [0 0.3], 'PositiveOverlapRange', [0.5 1]);
    
    save('rcnn-gtsdb.mat', 'rcnn');
else
    load('rcnn-gtsdb.mat', 'rcnn');
end