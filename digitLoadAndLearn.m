clear; clc;
mnist = fullfile('mnist');

allImages = imageDatastore(mnist,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');

shuffled = shuffle(allImages);

trainingSet = subset(shuffled,1:7000);
testSet = subset(shuffled,7001:10000);

img = readimage(trainingSet, 206);

[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);

cellSize = [4 4];
hogFeatureSize = length(hog_4x4);
numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');
w = waitbar(0, 'Wait...', 'Name', 'Training - Extracting HOG feature');
for i = 1:numImages
    clc;
    wt = sprintf('%i of %i', i, numImages);
    waitbar(i/numImages, w, wt);
    img = readimage(trainingSet, i);
    img = im2gray(img);
    
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end
close(w)

trainingLabels = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLabels);

[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);

predictedLabels = predict(classifier, testFeatures);

confMat = confusionmat(testLabels, predictedLabels);

helperDisplayConfusionMatrix(confMat)

function helperDisplayConfusionMatrix(confMat)
fprintf('Confusion Matrix');
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
digits = '0':'9';
colHeadings = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'digit  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '      |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end
end

function [features, setLabels] = helperExtractHOGFeaturesFromImageSet(imds, hogFeatureSize, cellSize)
setLabels = imds.Labels;
numImages = numel(imds.Files);
features  = zeros(numImages, hogFeatureSize, 'single');
w2 = waitbar(0, 'Wait...', 'Name', 'Test - Extracting HOG feature');
for j = 1:numImages
    clc;
    wt = sprintf('%i of %i', j, numImages);
    waitbar(j/numImages, w2, wt);
    
    img = readimage(imds, j);
    img = im2gray(img);
    
    img = imbinarize(img);
    
    features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
end
close(w2)
end