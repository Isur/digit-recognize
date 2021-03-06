[filename, pathname] = uigetfile('*.*', 'Select file');
filewithpath = strcat(pathname, filename);
image = imread(filewithpath);
imageP = imresize(image, [28 28]);
imageP = imbinarize(im2gray(imageP));
imshow(image);
[imgFeature, a] = extractHOGFeatures(imageP, 'CellSize', cellSize);
imshow(imageP);
p = predict(classifier, imgFeature);
title(char(p))
