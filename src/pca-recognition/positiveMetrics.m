function [TPs, TPTotal, FPs, FPTotal] = positiveMetrics(N)
% Calculate TPs, TPTotal, FPs, FPTotal

nTotal = sum(N(:));

% true positives
TPs = diag(N)' ./ sum(N, 1);
% total true positive
TPTotal = sum(diag(N)) ./ nTotal;
% false positives
FPs = (sum(N, 2) - diag(N))' ./ (nTotal - sum(N, 1));
% total false positives
FPTotal = (nTotal - sum(diag(N))) ./ ((size(N, 1) - 1) * nTotal);
end

