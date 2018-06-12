% Use test set of GTSDB to validate rcnn

nTestSet = size(testSet, 1);

pick = randsample(nTestSet, 1);

I = imread(testSet.imageFilename{pick});

% detect signs
[bboxes, scores, labels] = detect(rcnn, I, 'MiniBatchSize', 128);

outputImage = I;
% display detection results
for i = 1:length(scores)
    score = scores(i);
    label = labels(i);
    bbox = bboxes(i, :);
    
    annotation = sprintf('%s: (Confidence = %f)', label, score);
    
    outputImage = insertObjectAnnotation(outputImage, 'rectangle', bbox, annotation);
end

figure(1);
imshow(outputImage);