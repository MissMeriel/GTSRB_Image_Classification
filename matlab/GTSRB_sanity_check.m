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
%X = zeros(30, 30*(2*length(imagesArr1)+length(imagesArr2)));
X = zeros(30, 30*(2*length(imagesArr2)));
z = zeros(1, 30*(2*length(imagesArr2)));
for k = 1:length(imagesArr2)
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
    z(1+(k-1)*30:(k)*30) = 0;
end

offset = size(imagesArr2)*30;
offset = offset(1);
offsets = [1, offset];
datasetRootDir = 'GTSRB/Final_Training/Images/00001/'; %uigetdir; %gets current directory
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
for k = 1:length(imagesArr2)
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
    z(offset+1+(k-1)*30:offset+(k)*30) = 1;
end

lambda = 0.6;
Y = zeros(30, 30);
z_count = 0;
for m = 1:length(X)
    for l = 1:length(X)
        
        x_m = X(:, m);
        x_l = X(:, l);
       
        z2 = z(m);
        z1 = z(l);
        w_val = w_lm(z1, z2, lambda);
       
        Y = Y + w_val * (x_m - x_l) * (x_m - x_l)';
   end
end
whos Y
[V, D] = eigs(Y)

function lambda_val = w_lm(z1, z2, lambda)
    coeff = double(z1 == z2);
    lambda_val = ((1-lambda)^coeff) * ((1-lambda)^(1-coeff)); 
end