function features = featureExtractor(images)
% Feature extractor
% Input:    images      4-D matrix
% Output:   features    [nImages, nFeatures]

nImages = size(images, 4);
features = [];

for i = 1:nImages
    image = images(:,:,:,i);
    feature = extractHOGFeatures(image);
    features(end+1,:) = feature;
end

end

