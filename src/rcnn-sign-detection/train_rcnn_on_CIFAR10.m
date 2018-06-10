% Train the general R-CNN network model on CIFAR-10 data set.
% Requires R2016b or higher version.

addpath('utilities');


%% Download CIFAR-10 dataset (183MB)
% CIFAR-10 contains 50,000 training images

% add path if necessary
%addpath(fullfile(matlabroot,'examples','vision','main'));

cifar10Dir = 'CIFAR-10';

% try to download dataset
if exist(cifar10Dir, 'dir') == 7
    % exist() returns 7 if target is a directory
    
    % check if target directory is empty
    if length(dir(cifar10Dir)) == 2
        % any directory has '.' and '..' entries
        downloadCIFAR10Dataset(cifar10Dir);
    end
else
    mkdir(cifar10Dir);
    downloadCIFAR10Dataset(cifar10Dir);
end


%% Load training (50,000 32x32 RGB images) and test (10,000 32x32 RGB images) data
[trainingImages,trainingLabels,testImages,testLabels] = ...
    helperCIFAR10Data.load(cifar10Dir);

trainingCategories = categories(trainingLabels);
nCategories = size(trainingCategories,1);


%% Specify network architecture
% CNN architecture:
%   imageInputLayer
%   convolution2dLayer
%   reluLayer
%   maxPooling2dLayer
%   fullyConnectedLayer
%   softmaxLayer
%   classificationLayer


% Image input layer:
[height,width,nChannels,~] = size(trainingImages);

inputLayer = imageInputLayer([height,width,nChannels]);     % 32x32x3


% Middle layers (convolution + relu + pooling)s:

% convolution parameters
filterSize = [5 5];
nFilters = 32;

middleLayers = [
    % add padding to convolution kernels, to reduce information loss at the
    % image borders, especially in early layers
    convolution2dLayer(filterSize,nFilters,'Padding',2);
    % outputImageSize = 
    %   floor((inputImageSize - filterSize + 2*padding) / stride) + 1;
    % 32x32 -> 32x32
    
    % non-linear activation function: ReLu
    reluLayer();
    
    % 3x3 pooling with stride=2
    maxPooling2dLayer(3,'Stride',2);
    % 32x32 downsampled -> 15x15

    % repeat core layer-combination 3 times
    convolution2dLayer(filterSize,nFilters,'Padding',2);    % 15x15 -> 15x15
    reluLayer();
    maxPooling2dLayer(3,'Stride',2);    % 15x15 -> 7x7
    
    convolution2dLayer(filterSize,2 * nFilters,'Padding',2);    % 7x7 -> 7x7
    reluLayer();
    maxPooling2dLayer(3,'Stride',2);    % 7x7 -> 3x3
];


% Final layers (FC + relu + softmax)s:
finalLayers = [
    % fully connected layer with 64 output neurons
    fullyConnectedLayer(64);
    
    % ReLu
    reluLayer();
    
    % FC layer with [nCategories] output neurons
    fullyConnectedLayer(nCategories);
    
    softmaxLayer();
    classificationLayer();
];


% Cascade
layers = [
    inputLayer;
    middleLayers;
    finalLayers;
];


%% Initialize network weights (values of convolution kernels)
layers(2).Weights = 0.0001 * randn([filterSize nChannels nFilters]);


%% Train network on CIFAR-10
doTraining = false;     % re-train or load the pre-trained network

if doTraining
    % set training options
    options = trainingOptions('sgdm', ...
        'Momentum', 0.9, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 8, ...
        'L2Regularization', 0.004, ...
        'MaxEpochs', 40, ...
        'MiniBatchSize', 128, ...
        'Verbose', true);

    % start training
    cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, options);

    % save network model
    save('rcnn-cifar10.mat','cifar10Net');
else
    load('rcnn-cifar10.mat','cifar10Net');
end


%% Validate trained network
% visualize trained convolution kernels
% Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;

% rescale the weights to the range [0, 1] for better visualization
w = rescale(w);

figure;
montage(w);


