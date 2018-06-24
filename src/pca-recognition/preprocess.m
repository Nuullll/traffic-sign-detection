function [features] = preprocess(images)
% Preprocessing the loaded images, as a feature extractor
% Input:    images      [width,height,channels,nImages] 4-D uint8 matrix
% Output:   features    [nImages, nFeatures]

[w,h,~,nImages] = size(images);
% gray intensity as feature
nFeatures = w * h;
features = zeros(nImages, nFeatures);

% convert RGB to gray images
for i = 1:nImages
    image = images(:,:,:,i);
    grayImg = double(rgb2gray(image));
    % gray intensity as feature
    features(i,:) = grayImg(:)';
end

end

