function [negativeImages, negativeLabels] = loadNegative(dataDir, normSize)
% Load negative test images in data/
% The images will be resized to normSize

negativeRecipe = fullfile(dataDir, 'test_neg.txt');

negativeTable = readtable(negativeRecipe);

nNegative = height(negativeTable);

negativeImageFilenames = negativeTable.Var1;
negativeImageFilenames = cellfun(@(x) fullfile(dataDir, x), negativeImageFilenames, ...
    'UniformOutput', false);

% load negative images
negativeImages = uint8(zeros([normSize, 3, nNegative]));
for i = 1:nNegative
    filename = negativeImageFilenames{i};
    image = imread(filename);
    image = imresize(image, normSize);
    negativeImages(:,:,:,i) = image;
end

negativeLabels = categorical(cellfun(@str2double, negativeTable.Var2));

end

