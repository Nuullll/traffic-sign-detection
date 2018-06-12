% Use test set of GTSDB to validate rcnn

nTestSet = size(testSet, 1);

pick = randsample(nTestSet, 1);

I = imread(testSet.imageFilename{pick});

% detect signs
[bboxes, scores, labels] = detect(rcnn, I, 'MiniBatchSize', 128);

scoreThreshold = 0.9;
outputImage = I;
displayList = find(scores >= scoreThreshold);
% display detection results
for k = 1:length(displayList)
    i = displayList(k);
    score = scores(i);
    label = labels(i);
    bbox = bboxes(i, :);
    
    annotation = sprintf('%s: (Confidence = %f)', label, score);
    
    outputImage = insertObjectAnnotation(outputImage, 'rectangle', bbox, annotation);
end

figure(1);
imshow(outputImage);