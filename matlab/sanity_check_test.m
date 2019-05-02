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
dim = 30;
X1 = 1 * ones(dim,dim);
X2 = -1 * ones(dim,dim);
for i = 1:length(X1)
    for k = 1:length(X1)
        X1(i,k) = normrnd(1, 0.1); % X1(i,k) + rand*0.1;
        X2(i,k) = normrnd(-1, 0.1); % X2(i,k) + rand*0.1;
    end
end

X = [X1, X2];
disp('finished matrix concatenation')

lambda = 1;
Y = zeros(size(X1));
for m = 1:length(X)
    for l = 1:length(X)
        
        x_m = X(:, m);
        x_l = X(:, l);
       
        w_val = w_lm(m, l, lambda, dim);
       
        %Y = [Y, w_val * (x_m - x_l) * (x_m - x_l)'];
        Y = Y + w_val * (x_m - x_l) * (x_m - x_l)';
    end
    if(mod(m, 10))
        disp('finished column processing for col '+string(m))
    end
end

whos Y
[V, D] = eigs(Y)
D_len = size(D);
D_len = D_len(2);
V_min = V(:,D_len-1:D_len);
mins = min(D)
if(mins(1) < 0)
    V_min = V(:,1:2);
end
coords = V_min' * X;

hold on
scatter(coords(1, 1:dim), coords(2, 1:dim), 'filled', 'g')
scatter(coords(1, dim+1:dim*2), coords(2, dim+1:dim*2), 'filled','b')
save('sanity_check_test_output_100.mat')
toc


function lambda_val = w_lm(z1, z2, lambda, dim)
    coeff = double(z1 < dim+1 && z2 < dim+1);
    lambda_val = ((1-lambda)^coeff) * ((-lambda)^(1-coeff)); 
end