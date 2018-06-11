% Fine-tune the pretrained CNN (on CIFAR-10) for traffic sign detection

%% Load GTSDB dataset (900 images with ROI labels)
gtsdbDir = fullfile(pwd, 'Origin\');
gtsdbData = loadGTSDB2Table(gtsdbDir);