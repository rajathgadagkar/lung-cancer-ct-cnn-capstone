%% Lung CT Scan Image Pre-Processing & Marker-Controlled Watershed Segmentation
% Official B.Tech Capstone Project — PES University, Bengaluru (Dept. of ECE)
% Title : Lung Cancer Detection and Classification using Image Processing and CNN
% Author: Rajath Gadagkar <rajathgadagkar@gmail.com>
%         (with Rhitesh Kumar Singh, Dhruv K.C, Nitin Kumar Pandit)
% Guide : Prof. H R Vanamala
%
% Usage:
%   preprocess_watershed('data/raw_dicom', 'data/preprocessed_dicom');

function preprocess_watershed(inputDir, outputDir)
    if nargin < 1 || isempty(inputDir)
        inputDir = fullfile('data', 'raw_dicom');
    end
    if nargin < 2 || isempty(outputDir)
        outputDir = fullfile('data', 'preprocessed_dicom');
    end

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    dicomList = dir(fullfile(inputDir, '*.dcm'));
    fprintf('Found %d DICOM slices in %s\n', numel(dicomList), inputDir);

    for cnt = 1:numel(dicomList)
        fileName = dicomList(cnt).name;
        fprintf('[%d/%d] Processing %s\n', cnt, numel(dicomList), fileName);

        % 1. Read DICOM CT slice
        img = dicomread(fullfile(inputDir, fileName));

        % 2. Noise removal: Median filter (salt & pepper) + Gaussian filter (sigma = 2)
        medianImg = medfilt2(img);
        gaussImg  = imgaussfilt(medianImg, 2);

        % 3. Local contrast enhancement via Adaptive Histogram Equalization (CLAHE)
        enhancedImg = adapthisteq(mat2gray(gaussImg));

        % 4. Global Otsu thresholding
        level = graythresh(enhancedImg);
        bw    = imbinarize(enhancedImg, level);

        % 5. Complement + negative Euclidean distance transform
        distMap = -bwdist(~bw);
        distMap(~bw) = -Inf;

        % 6. Extended-minima filtering & minima imposition to prevent over-segmentation
        mask = imextendedmin(distMap, 2);
        distImposed = imimposemin(distMap, mask);

        % 7. Watershed segmentation & boundary superposition
        labels = watershed(distImposed);
        bwWatershed = bw;
        bwWatershed(labels == 0) = 0;

        % 8. Difference composite fusion and 512x512 resizing
        fused   = imfuse(bwWatershed, enhancedImg, 'diff');
        outImg  = imresize(im2double(fused), [512 512]);

        % 9. Write preprocessed DICOM slice
        dicomwrite(outImg, fullfile(outputDir, fileName));
    end

    fprintf('Completed preprocessing %d DICOM slices -> %s\n', numel(dicomList), outputDir);
end
