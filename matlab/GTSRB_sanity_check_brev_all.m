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
clear
tic
pwd
dim = 30;
sample_sz = 55;
sample_period = 5;
show = true
datasetRootDir = 'GTSRB/Final_Training/Images/00001/'; %uigetdir; %gets current directory
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
X1 = zeros(dim, dim*sample_sz);
z = ones(1, dim*sample_sz);
for k = 1:sample_sz
    filename = strcat(datasetRootDir, imagesArr(k*sample_period).name);
    temp = imread(filename);
    temp = rgb2gray(temp);
    %info = imfinfo(filename);
    %disp(info);
    %imshow(temp)
    bigImage = imresize(temp, [dim,dim]);
    %whos bigImage
    if(show)
        imshow(imresize(temp, [100,100]))
    end
    %iminfo(bigImage)
    X1(:, (dim*(k-1))+1:(dim*k)) = bigImage;
end

datasetRootDir = 'GTSRB/Final_Training/Images/00003/'; %uigetdir; %gets current directory
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
X2 = zeros(dim,dim*sample_sz);
for k = 1:sample_sz
    z = [z, 2*ones(1,dim)];
    filename = strcat(datasetRootDir, imagesArr(k*sample_period).name);
    temp = imread(filename);
    imshow(imresize(temp, [100,100]))
    temp = rgb2gray(temp);
    %whos temp
    %imshow(temp)
    %imfinfo(filename);
    bigImage = imresize(temp, [dim,dim]);
    %whos bigImage
    if(show)
        imshow(imresize(temp, [100,100]))
    end
    %iminfo(bigImage)
    X2(:,(dim*(k-1))+1:(dim*k)) = bigImage;
end
X = [X1, X2];
disp('finished matrix concatenation')
lambda = 0.51;
Y = zeros(dim, dim);
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
    if(mod(m, 500) == 0)
        %disp('finished column processing for col '+string(m))
        disp('% finished: '+string(double(m)/length(X)))
    end
end

whos Y
[V, D] = eigs(Y)
D_len = size(D);
D_len = D_len(2);
V_min = V(:,D_len-2:D_len);
mins = min(D);
if(mins(1) < 0)
    V_min = V(:,1:3);
end
coords = V_min' * X;
save('sanity_check_brev_3Doutput_051_55sample.mat')
figure
hold on
scatter3(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), coords(3, 1:dim*sample_sz), 'filled', 'g')
scatter3(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), coords(3, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')

toc

function lambda_val = w_lm(z1, z2, lambda)
    coeff = double(z1 == z2);
    lambda_val = ((1-lambda)^coeff) * ((-lambda)^(1-coeff)); 
end