%% 1. Image Binarization
img = imread('C:\Users\ziada\OneDrive\Desktop\');
grayImg = rgb2gray(img);
binaryImg = imbinarize(grayImg);

figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(binaryImg);
title('Binarized Image');

%% 2. Connected Components and Region Segmentation
img = imread('C:\Users\ziada\OneDrive\Desktop\Ziad\Master\Term 1\');
grayImg = rgb2gray(img);
binaryImg = imbinarize(grayImg);
cc = bwconncomp(binaryImg);
stats = regionprops(cc, 'BoundingBox');

figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');
subplot(1, 2, 2);
imshow(img);
hold on;
for k = 1:length(stats)
    thisBB = stats(k).BoundingBox;
    rectangle('Position', thisBB, 'EdgeColor', 'r', 'LineWidth', 2);
end
title('Segmented Regions');
hold off;

%% 3. Edge Detection using Canny
img = imread('C:\Users\ziada\OneDrive\Desktop\');
edges = edge(rgb2gray(img), 'canny');

figure;
subplot(1,2,1);
imshow(img);
title('Original Image');

subplot(1,2,2);
imshow(edges);
title('Edge Detection Result');

%% 4. Salt-and-Pepper Noise Removal using Median Filter
img = imread('C:\Users\ziada\OneDrive\Desktop\');

gray_img = rgb2gray(img);

sp_filtered_img = medfilt2(gray_img, [3 3]);

figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(sp_filtered_img);
title('Salt-and-Pepper Noise Removed');

%% 5. Hough Transform for Line Detection
imagePath = 'C:\Users\ziada\OneDrive\Desktop\Ziad\Master\';
img = imread(imagePath);

gray_img = rgb2gray(img);

edges = edge(gray_img, 'Sobel');

[H, theta, rho] = hough(edges);

peaks = houghpeaks(H, 10);  % Top 10 most prominent lines
lines = houghlines(edges, theta, rho, peaks);

imshow(img);
hold on;

for k = 1:length(lines)
    % Coordinates of the line start and endpoints
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1), xy(:,2), 'LineWidth', 2, 'Color', 'red');
end

title('Text Line Recognition');
hold off;

%% 6. Morphological Thinning
imagePath = 'C:\Users\ziada\OneDrive\';

img = imread(imagePath);
if numel(unique(img)) > 2
    img = imbinarize(rgb2gray(img));
end

thinned_img = bwmorph(img, 'thin', Inf);

figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(thinned_img);
title('Thinned Image');

%% 7. Histogram Equalization
imagePath = 'C:\Users\ziada\OneDrive\';

img = imread(imagePath);

if size(img, 3) == 3
    gray_img = rgb2gray(img);
else
    gray_img = img;
end

eq_img = histeq(gray_img);

figure;
subplot(1, 2, 1);
imshow(gray_img);
title('Original Image');

subplot(1, 2, 2);
imshow(eq_img);
title('Histogram Equalized Image');

%% 8. Gaussian Smoothing/Filtering
imagePath = 'C:\Users\ziada\OneDrive\Desktop\Ziad\Master\Term 1\Image';

img = imread(imagePath);

if size(img, 3) == 3
    gray_img = rgb2gray(img);
else
    gray_img = img;
end

kernel_size = 5;
sigma = 1.5;

smoothed_img = imgaussfilt(gray_img, sigma, 'FilterSize', kernel_size);

figure;
subplot(1, 2, 1);
imshow(gray_img);
title('Original Image');

subplot(1, 2, 2);
imshow(smoothed_img);
title('Smoothed Image');

%% 9. SURF Feature Detection
imagePath = 'C:\Users\ziada\OneDrive\Desktop\Ziad\Master\Tr';

img = imread(imagePath);
gray_img = rgb2gray(img);

points = detectSURFFeatures(gray_img);
[features, validPoints] = extractFeatures(gray_img, points);

figure; imshow(img);
hold on;
plot(validPoints.selectStrongest(50), 'showOrientation', true);
title('Feature Points');

%% 10. Morphological Operations and Object Detection
imagePath = 'C:\Users\ziada\OneDrive\Desktop\Ziad\Master\Term 1\image processing';

img = imread(imagePath);
gray_img = rgb2gray(img);
binary_img = imbinarize(gray_img);
binary_img = ~binary_img;

se = strel('rectangle', [5, 5]);
dilated_img = imdilate(binary_img, se);

[labels, num] = bwlabel(dilated_img);
stats = regionprops(labels, 'BoundingBox');

imshow(img);
hold on;
for k = 1:numel(stats)
    rectangle('Position', stats(k).BoundingBox, 'EdgeColor', 'r', 'LineWidth', 1);
end