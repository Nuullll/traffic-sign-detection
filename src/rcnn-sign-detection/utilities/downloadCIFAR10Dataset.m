function downloadCIFAR10Dataset( dirname )
% Download CIFAR-10 dataset (183MB) to the target directory
% Please make sure the directory already exists

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,dirname);

end

