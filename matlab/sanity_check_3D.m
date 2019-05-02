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
sample_sz = 120;
sample_period = 5;
show = false

datasetRootDir = 'GTSRB/Final_Training/Images/00001/'
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
size(imagesArr)

X1 = zeros(dim*dim*3, sample_sz);
z = ones(1, sample_sz);
for k = 1:sample_sz
    filename = strcat(datasetRootDir, imagesArr(k*sample_period).name);
    temp = imread(filename);
    bigImage = imresize(temp, [dim,dim], 'bicubic');
    %whos bigImage
    %iminfo(bigImage)
    if(show)
        imshow(imresize(temp, [100,100]))
    end
    % vectorize image tensor by slab
    vec_image = vectorize_image(bigImage);
    X1(:, k) = vec_image;
end

datasetRootDir = 'GTSRB/Final_Training/Images/00003/';
% get all .ppm files in struct
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); 
X2 = zeros(dim*dim*3,sample_sz);
for k = 1:sample_sz
    z = [z, 2*ones(1,sample_sz)];
    filename = strcat(datasetRootDir, imagesArr(k*sample_period).name);
    temp = imread(filename);
    bigImage = imresize(temp, [dim,dim], 'bicubic');
    %whos bigImage
    %iminfo(bigImage)
    if(show)
        imshow(imresize(temp, [100,100]))
    end
    vec_image = vectorize_image(bigImage);
    X2(:, k) = vec_image;
end
X = [X1, X2];
disp('finished matrix concatenation')

lambda = 0.50;
Y = zeros(dim*dim*3,dim*dim*3);
z_count = 0;
[X_I, X_J]=size(X);
for m = 1:X_J
    for l = 1:X_J
        
        x_m = X(:, m);
        x_l = X(:, l);
       
        z2 = z(m);
        z1 = z(l);
        w_val = w_lm(z1, z2, lambda);
        temp = w_val * (x_m - x_l) * (x_m - x_l)';
        Y = Y + temp;
    end
    if(mod(m, 500) == 0)
        %disp('finished column processing for col '+string(m))
        disp('% finished: '+string(double(m)/length(X)))
    end
end

whos Y
[V, D] = eigs(Y);
D_len = size(D);
D_len = D_len(2);
%V_min = V(:,D_len-1:D_len);
V_min = V(:,D_len-2:D_len);
mins = min(D);
if(mins(1) < 0)
    %V_min = V(:,1:2);
    V_min = V(:,1:3);
end
coords = V_min' * X;
save('sanity_check_scatter_3D_fullvec_output_050_120sample.mat')

hold on
scatter3(coords(1, 1:sample_sz), coords(2, 1:sample_sz), coords(3, 1:sample_sz), 'filled', 'g')
scatter3(coords(1, sample_sz+1:sample_sz*2), coords(2, sample_sz+1:sample_sz*2), coords(3, 3*sample_sz+1:3*sample_sz*2), 'filled','b')

toc

function vec_image = vectorize_image(img)
    [I,J,K]=size(img);
    vec_image = []; %zeros(I*J,K); % [];
    for k = 1:K
        vec_image = [vec_image; vec(img(:,:,k))];
    end
end

function v = vec(A)
    sz = size(A);
    if(length(sz) > 2)
        error('Error in vec - This function can only be applied to matrices')
    end
    v = zeros(sz(1) * sz(2), 1);
    index = 1;
    for col = 1:sz(2)
        for row = 1:sz(1)
            v(index, 1) = A(row, col);
            index=index+1;
        end
    end
end


function lambda_val = w_lm(z1, z2, lambda)
    coeff = double(z1 == z2);
    lambda_val = ((1-lambda)^coeff) * ((-lambda)^(1-coeff)); 
end