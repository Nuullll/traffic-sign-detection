function [coeff, score, mu] = pcaEnergy(data, threshold)
% Select principle components according to the energy threshold.
% Input:    data        [nObservations, nFeatures]
% Input:    threshold   energy threshold to select principle components
% Output:   coeff       eigen matrix for the pca
% Output:   score       low-d data representations
% Output:   mu          mean of the sample data

[coeff, score, latent, ~, ~, mu] = pca(data);

% find threshold
c = sum(latent);
energy = cumsum(latent/c);
nComponents = find(energy >= threshold, 1, 'first');

coeff = coeff(:, 1:nComponents);
score = score(:, 1:nComponents);
end

