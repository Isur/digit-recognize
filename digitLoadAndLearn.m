clear; clc;
syntheticDir   = fullfile(toolboxdir('vision'), 'visiondata','digits','synthetic');
handwrittenDir = fullfile(toolboxdir('vision'), 'visiondata','digits','handwritten');

trainingSet = imageDatastore(syntheticDir,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(handwrittenDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

img = readimage(trainingSet, 206);

[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);

cellSize = [4 4];
hogFeatureSize = length(hog_4x4);
numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    
    img = rgb2gray(img);
    
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

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

for j = 1:numImages
    img = readimage(imds, j);
    img = rgb2gray(img);
    
    img = imbinarize(img);
    
    features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
end
end