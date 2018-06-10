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


%% Load training (50,000 RGB images) and test (10,000 RGB images) data
[trainingImages,trainingLabels,testImages,testLabels] = ...
    helperCIFAR10Data.load(cifar10Dir);

trainingCategories = categories(trainingLabels);


%% Training R-CNN on CIFAR-10
