% Train the general R-CNN network model on CIFAR-10 data set.
% Requires R2016b or higher version.

%% Download CIFAR-10 dataset
% CIFAR-10 contains 50,000 training images

% add path if necessary
addpath(fullfile(matlabroot,'examples','vision','main'));

% check directory
dirname = 'CIFAR-10';
if exist(dirname, 'dir') == 7
    
end