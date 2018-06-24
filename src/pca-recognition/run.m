% compare performances under different parameters

%% Define parameters
normSize = [25 25];


%% Run pca+knn model
for K = 1:2:5
    n = 20;
    thresholdList = linspace(0,1,n);
    TPList = zeros(n,1);
    FPList = zeros(n,1);
    for i = 1:n
        energyThreshold = thresholdList(i);
        [N, TPs, TPTotal, FPs, FPTotal] = pcaKnnModel(normSize, energyThreshold, K);
        TPList(i) = TPTotal;
        FPList(i) = FPTotal;
    end
    
    figure(1);
    hold on;
    plot(thresholdList, TPList);
    
    figure(2);
    hold on;
    plot(thresholdList, FPList);
end


%% Captions
figure(1);
legend('K=1','K=3','K=5');
xlabel('PCA energy');
ylabel('TP total');

figure(2);
legend('K=1','K=3','K=5');
xlabel('PCA energy');
ylabel('FP total');
