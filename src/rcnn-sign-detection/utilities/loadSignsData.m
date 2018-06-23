function [images,labels] = loadSignsData(signsDataDir)
% Load (pure) sign images in signsDataDir
% WARNING: all the loaded images will be resized to 32x32
% Output:
% images    4-D     matrix
% labels    nx1     categorical

normSize = [32 32];
images = [];
labels = [];

% map detailed categories into 4 categories
categoryMap = [0;0;0;0;0;0;3;0;0;0;0;1;3;3;3;0;0;3;1;1;1;1;1;1;1;1;1;1;...
    1;1;1;1;3;2;2;2;2;2;2;2;2;3;3];

% Traverse all subfolders to get images
% folder name as label
D = dir(signsDataDir);

for i = 3:length(D)     % skip '.' and '..'
    item = D(i);
    
    % skip files
    if ~item.isdir
        continue
    end
    
    label = categoryMap(str2double(item.name)+1);
    subpath = fullfile(signsDataDir, item.name);
    
    imageFiles = dir(subpath);
    
    for j = 3:length(imageFiles)
        display(imageFiles(j).name);
        image = imread(fullfile(subpath, imageFiles(j).name));
        image = imresize(image, normSize);
        images(:,:,:,end+1) = image;
        labels(end+1) = label;
    end
end
    
end

