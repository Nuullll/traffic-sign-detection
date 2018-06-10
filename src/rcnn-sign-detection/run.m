% traffic sign detection using R-CNN
% https://www.mathworks.com/help/vision/examples/object-detection-using-deep-learning.html
% 
% R-CNN is an object detection framework,
% which uses a CNN to classify image REGIONS within an image,
% instead of using a sliding window.
%
% Transfer learning: 
% Use a pretrained network (which has already learned a rich set of image
% features) for the new task by fine-tuning the network.
% A network is fine-tuned by making small adjustments to the weights 
% such that the feature representations learned for the original task 
% (general object detection) are slightly adjusted to support the new task 
% (traffic sign detection).
% Transfer learning reduces the number of the required training images 
% and the time of the training process.


% Train R-CNN on CIFAR-10
train_cnn_on_CIFAR10;