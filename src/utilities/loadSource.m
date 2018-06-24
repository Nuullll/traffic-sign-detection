function [trainImages, trainLabels, testImages, testLabels] = loadSource(sourceDir, normSize, trainRatio)
% Load images in source/
% The images will be resized to normSize
% The images will be divided into trainImages and testImages according to
% trainRatio.

% project 43 detailed categories into 4 main categories
categoryMap = [0;0;0;0;0;0;3;0;0;0;0;1;3;3;3;0;0;3;1;1;1;1;1;1;1;1;1;1;...
    1;1;1;1;3;2;2;2;2;2;2;2;2;3;3];

nSource = 1127;
images = uint8(zeros([normSize, 3, nSource]));
labels = zeros(nSource, 1);

D = dir(sourceDir);
n = 1;

for i = 3:length(D)
    name = D(i).name;
    if D(i).isdir
        label = categoryMap(str2double(name)+1);
        T = dir(fullfile(sourceDir, name, '*.bmp'));
        for j = 3:length(T)
            filename = fullfile(T(j).folder, T(j).name);
            image = imread(filename);
            image = imresize(image, normSize);
            images(:,:,:,n) = image;
            labels(n) = label;
            n = n + 1;
        end
    end
end


% generate train set and test set
trainI = [];
for cat = 0:3
    I = find(labels == cat);
    num = length(I);
    nTrain = floor(trainRatio * num);
    trainI = [trainI; randsample(I, nTrain)];
end

testI = setdiff(1:nSource, trainI);

trainImages = images(:,:,:,trainI);
trainLabels = categorical(labels(trainI));
testImages = images(:,:,:,testI);
testLabels = categorical(labels(testI));
    

end

