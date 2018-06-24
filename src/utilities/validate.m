function N = validate(targetLabels, predictedLabels)
% Validate the correctness of predicted labels
% Input:    targetLabels        ground truth
% Input:    predictedLabels     predicted labels
% Output:   N(i,j)              number of j(ground truth) predicted as i

categories = unique(targetLabels);
nCategories = length(categories);

N = zeros(nCategories);

for j = 1:nCategories
    category = categories(j);
    I = find(targetLabels == category);
    labels = predictedLabels(I);
    for i = 1:nCategories
        predictedCat = categories(i);
        N(i,j) = sum(labels == predictedCat);
    end
end

end

