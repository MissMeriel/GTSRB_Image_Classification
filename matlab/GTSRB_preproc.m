% HOSVD for my data set
% Reference: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset#Downloads

% Read each road sign in directory into Matlab
% Convert to grayscale
% Add as tensor slab
%X = double(imread('/Users/Meriel/Documents/UVA/Tensors4DataSci/HW2/GTSRB/Final_Training/Images/00000/00000_00000.ppm', 'ppm'));
% We're doing this for one sign so we don't run out of memory. For everything in directory:
% HW2/GTSRB_subset/Final_Training/Images/00002 30kph sign, 266 images
% HW2/GTSRB_subset/Final_Training/Images/00002 50kph sign, 17 images
% HW2/GTSRB_subset/Final_Training/Images/00003 60kph sign, 316 images

% Focal length of camera:
%   Sensor has 11mm diagonal with 3:4 aspect ratio --> sensor is 8.8 x 6.6 mm
%   Images are 1024 x 1360 px --> 270.9 x 359.8 mm
%   1360/8.8 = 154.5 focal length 
%   
%   Sensor specs: https://www.1stvision.com/cameras/sensor_specs/ICX285.pdf
pwd
datasetRootDir = 'GTSRB/Final_Training/Images/00001/'; %uigetdir; %gets current directory
imagesArr1 = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
datasetRootDir2 = 'GTSRB/Final_Training/Images/00003/'; %uigetdir; %gets current directory
imagesArr2 = dir(fullfile(datasetRootDir2,'*.ppm'));
X = zeros(30, 30*(length(imagesArr1)+length(imagesArr2)));
for k = 1:length(imagesArr1)
    filename = strcat(datasetRootDir, imagesArr1(k).name);
    temp = imread(filename);
    temp = rgb2gray(temp);
    %info = imfinfo(filename);
    %disp(info);
    %width(k) = info.Width;
    %height(k) = info.Height;
    %whos temp
    %imshow(temp)
    bigImage = imresize(temp, [30,30]);
    %whos bigImage
    %imshow(bigImage)
    %iminfo(bigImage)
    X(:,1+(k-1)*30:(k)*30) = bigImage;
end

offset = size(imagesArr1)
offset = offset(1)
datasetRootDir = 'GTSRB/Final_Training/Images/00001/'; %uigetdir; %gets current directory
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
for k = 1:length(imagesArr)
    filename = strcat(datasetRootDir, imagesArr(k).name);
    temp = imread(filename);
    temp = rgb2gray(temp);
    %info = imfinfo(filename);
    %whos temp
    %imshow(temp)
    bigImage = imresize(temp, [30,30]);
    %whos bigImage
    %imshow(bigImage)
    %iminfo(bigImage)
    X(:,offset+1+(k-1)*30:offset+(k)*30) = bigImage;
end

