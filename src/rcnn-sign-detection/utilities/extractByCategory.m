function [categoryData,dataInd] = extractByCategory(datasetTable,categoryName)
% Extract a single category data from the whole dataset.

C = datasetTable.(categoryName);
dataInd = find(~cellfun(@isempty, C));
categoryData = C(dataInd);

end

