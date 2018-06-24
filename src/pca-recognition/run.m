% compare performances under different parameters

%% Define parameters
normSize = [25 25];
distanceThreshold = Inf;
doOpenTest = false;

%% Run pca+knn model
for K = 1:2:5
    n = 20;
    thresholdList = linspace(0,1,n);
    TPList = zeros(n,1);
    FPList = zeros(n,1);
    for i = 1:n
        energyThreshold = thresholdList(i);
        [N, TPs, TPTotal, FPs, FPTotal] = pcaKnnModel(normSize, energyThreshold, K, distanceThreshold, doOpenTest);
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


%% Open test
normSize = [25 25];
energyThreshold = 0.9;
K = 1;
distanceThreshold = 0.02;
doOpenTest = true;

n = 100;
distanceThresholdList = linspace(0.005,0.055,n);

TPList = zeros(n,1);
FPList = zeros(n,1);
for i = 1:n
    distanceThreshold = distanceThresholdList(i);
    [N, TPs, TPTotal, FPs, FPTotal] = pcaKnnModel(normSize, energyThreshold, K, distanceThreshold, doOpenTest);
    TPList(i) = TPTotal;
    FPList(i) = FPTotal;
end

figure(3);
plot(distanceThresholdList, TPList);

figure(4);
plot(distanceThresholdList, FPList);

%% Captions
figure(3);
xlabel('Distance Threshold');
ylabel('TP total');

figure(4);
xlabel('Distance Threshold');
ylabel('FP total');
