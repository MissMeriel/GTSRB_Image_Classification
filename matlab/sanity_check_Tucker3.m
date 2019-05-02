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

% perform classification
% find measure o fseparation
% - rate of misclassificaiton
% - compare inter vs intra class variablility (signal vs noise)
% - run support vector machine or kernel SVM

% mode feature choice
% R + G + B (color)
% grayscale + color histogram + original resolution
% grayscale constrained to ROI + 

% algorithms
% - linear subspace learning
% - Tucker ml subspace learning
% - truncated HOSVD subspace learning
% - truncated HOSVD subspace learning with support vector machine
% - truncated HOSVD subspace learning with kernel methods


clear 

% generate X with vectorized tensors as columns
dim = 20;
dims = [dim,dim,3];

X_imode = [];
X_jmode = [];
X_kmode = [];
z = [];
tic
pwd
sample_sz = 100;
sample_period = 5;
show = false

datasetRootDir = 'GTSRB/Final_Training/Images/00001/'
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
size(imagesArr)

datasetRootDir = 'GTSRB/Final_Training/Images/00002/'
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
size(imagesArr)

datasetRootDir = 'GTSRB/Final_Training/Images/00003/'
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
size(imagesArr)

datasetRootDir = 'GTSRB/Final_Training/Images/00004/'
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
size(imagesArr)

datasetRootDir = 'GTSRB/Final_Training/Images/00005/'
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
size(imagesArr)

datasetRootDir = 'GTSRB/Final_Training/Images/00007/'
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
size(imagesArr)

X1 = zeros(dims(1)*dims(2)*dims(3), sample_sz);
z = ones(1, sample_sz);
for k = 1:sample_sz
    filename = strcat(datasetRootDir, imagesArr(k*sample_period).name);
    temp = imread(filename);
    bigImage = imresize(temp, [dim,dim], 'bicubic');
    bigImage = im2double(bigImage);
    %whos bigImage
    %iminfo(bigImage)
    if(show)
        imshow(imresize(temp, [100,100]))
    end
    % vectorize image tensor by slab
    vec_image = vectorize_image(bigImage);
    X1(:, k) = vec_image;
    test_tensor = bigImage;
    X_imode = [X_imode, vectorize_tensor_imode(bigImage)];
    X_jmode = [X_jmode, vectorize_tensor_jmode(bigImage)];
    X_kmode = [X_kmode, vectorize_tensor_kmode(bigImage)];
end

datasetRootDir = 'GTSRB/Final_Training/Images/00003/';
% get all .ppm files in struct
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); 
X2 = zeros(dim*dim*3,sample_sz);
for k = 1:sample_sz
    z = [z, 2];
    filename = strcat(datasetRootDir, imagesArr(k*sample_period).name);
    temp = imread(filename);
    bigImage = imresize(temp, [dim,dim], 'bicubic');
    bigImage = im2double(bigImage);
    %whos bigImage
    %iminfo(bigImage)
    if(show)
        imshow(imresize(temp, [100,100]))
    end
    vec_image = vectorize_image(bigImage);
    X2(:, k) = vec_image;
    %vectorize_tensor_imode(X1)
    X_imode = [X_imode, vectorize_tensor_imode(bigImage)];
    X_jmode = [X_jmode, vectorize_tensor_jmode(bigImage)];
    X_kmode = [X_kmode, vectorize_tensor_kmode(bigImage)];
end
disp('finished matrix concatenation');
X1 = bigImage;
[I,J,K] = size(bigImage);

r1 = 1;
r2 = 1;
r3 = 3;
[x1, x2, x3] = unfold(X1);
[x1_1, x2_1, x3_1] = unfold(test_tensor);

% initialize U
U_init = zeros(I, I);
sz = size(x1');
x1T = x1';
for i = 1:sz(2)
    U_init = U_init + x1T(:,i) * x1T(:,i)';
end
[U,e] = eigs(U_init);
U = U(:,1:r1);

% initialize V
V_init = zeros(J, J);
x2T = x2';
sz = size(x2T);
for i = 1:sz(1)
    V_init = V_init + x2T(:,i) * x2T(:,i)';
end
[V,e] = eigs(V_init);
V = V(:,1:r1);

% initialize W
W_init = zeros(K, K);
x3T = x3';
sz = size(x3T);
for i = 1:sz(1)
    W_init = W_init + x3T(:,i) * x3T(:,i)';
end
[W,e] = eigs(W_init);
W = W(:,1:r1);

lambda = 1 % 0.99999999999999990;
Y1 = zeros(dims(1)*dims(2)*dims(3), dims(1)*dims(2)*dims(3));

[X_I,X_J] = size(X_imode);
for m = 1:X_J
    for l = 1:X_J
        
        x_m = X_imode(:, m);
        x_l = X_imode(:, l);
       
        z2 = z(m);
        z1 = z(l);
        w_val = w_lm(z1, z2, lambda);
        temp = w_val * (x_m - x_l) * (x_m - x_l)';
        Y1 = Y1 + temp;
    end
end

Y2 = zeros(dims(1)*dims(2)*dims(3), dims(1)*dims(2)*dims(3));
[X_I,X_J] = size(X_jmode);
for m = 1:X_J
    for l = 1:X_J
        
        x_m = X_jmode(:, m);
        x_l = X_jmode(:, l);
       
        z2 = z(m);
        z1 = z(l);
        w_val = w_lm(z1, z2, lambda);
        temp = w_val * (x_m - x_l) * (x_m - x_l)';
        Y2 = Y2 + temp;
    end
end

Y3 = zeros(dims(1)*dims(2)*dims(3), dims(1)*dims(2)*dims(3));
[X_I,X_J] = size(X_kmode);
for m = 1:X_J
    for l = 1:X_J
        
        x_m = X_kmode(:, m);
        x_l = X_kmode(:, l);
       
        z2 = z(m);
        z1 = z(l);
        w_val = w_lm(z1, z2, lambda);
        temp = w_val * (x_m - x_l) * (x_m - x_l)';
        Y3 = Y3 + temp;
    end
end

% clear memory
%clear bigImage imagesArr temp vec_image x1T x2T x3T U_init V_init W_init 

% _|_Tucker3 test
n = 0;
rel_tucker_errors = [];
prev_rel_tucker_error=10;
rel_tucker_error = 1;
signal2=0;
error_delta = 1;
figure
while ((abs(error_delta) > 0.0001) && (n < 50))
    n = n + 1;
    [I,J,K] = size(X1);
    % calc Zs
    L = kron(U*U', V*V');
    Z1 = zeros(K, K);
    for ij = 1:I*J
        offset1 = ij*K;
        end_index1 = offset1 - K + 1;
        for ij2 =  1:I*J
            offset2 = ij2*K;
            end_index2 = offset2 - K + 1;
            %disp('size of Y3: '+string(size(Y3)));
            %L(ij, ij2)
            Y3(end_index1:offset1, end_index2:offset2);
            temp= L(ij, ij2) * Y1(end_index1:offset1, end_index2:offset2);
            Z1 = Z1 + temp;
        end
    end
    
    L = kron(U*U', W*W');
    Z2 = zeros(J, J);
    for row = 1:I*K
        offset1 = row*J;
        end_index1 = offset1 - J + 1;
        for col =  1:I*K
            offset2 = col*J;
            end_index2 = offset2 - J + 1;
            temp= L(row, col) * Y1(end_index1:offset1, end_index2:offset2);
            Z2 = Z2 + temp;
        end
    end
    
    L = kron(W*W', V*V');
    Z3 = zeros(I, I);
    for row = 1:J*K
        offset1 = row*I;
        end_index1 = offset1 - I + 1;
        for col =  1:J*K
            offset2 = col*I;
            end_index2 = offset2 - I + 1;
            temp= L(row, col) * Y1(end_index1:offset1, end_index2:offset2);
            Z3 = Z3 + temp;
        end
    end
    
    % update W
    [V_min, D] = eigs(Z1);
    D_len = size(D);
    D_len = D_len(2);
    V_min = V_min(:,D_len-2:D_len);
    mins = min(D);
    if(mins(1) < 0)
        V_min = V(:,1:3);
    end
    %W
    W = V_min(:, 1:r1);
    
    
    % update V
    [V_min, D] = eigs(Z2);
    D_len = size(D);
    D_len = D_len(2);
    V_min = V_min(:,D_len-2:D_len);
    mins = min(D);
    if(mins(1) < 0)
        V_min = V(:,1:3);
    end
    %V
    V = V_min(:, 1:r2);
    
    % update U
    [V_min, D] = eigs(Z3);
    D_len = size(D);
    D_len = D_len(2);
    V_min = V_min(:,D_len-2:D_len);
    mins = min(D);
    if(mins(1) < 0)
        V_min = V(:,1:3);
    end
    %U
    U = V_min(:, 1:r3);

    %calc G
    n
    %g = kron(kron(U, V), W)' * Y3;    
    G3 = kron(V,U)' * x3_1 * W;
    G = reshape(G3, [r1 r2 r3]);

    %calc Xm (aka X~)
    %Xm = kron(kron(U, V), W) * g;
    X3 = kron(V,U) * G3 * W';
    Xm = reshape(X3, size(test_tensor));

    
    %calc error & convergence
    error2=0;
    estimateNorm = 0;
    [rows, cols] = size(X1);
    for k=1:K
        X1(:,:,k)
        signal2=signal2+norm(X1(:,:,k),'fro')^2;
        X1(:,:,k)-Xm(:,:,k)
        error2=error2+norm(X1(:,:,k)-Xm(:,:,k),'fro')^2;
        estimateNorm = estimateNorm+norm(Xm(:,:,k),'fro')^2;
    end
    rel_tucker_error = error2/signal2;
    rel_tucker_errors(n) =  rel_tucker_error;
    error_delta = prev_rel_tucker_error - rel_tucker_error;
    prev_rel_tucker_error = rel_tucker_error;

    %plot relative tucker error
    hold on
    pl = plot(rel_tucker_errors, 'g');
    title('\fontsize{24}GTSRB Data Subset Decomposition');
    xlabel('\fontsize{16}Steps')
    ylabel('\fontsize{16}Relative Tucker error')
    pause(0);
end

% compute g's
size(U)
size(V)
size(W)
size(kron(kron(U, V), W))
[rows, cols] = size(X_imode)
coords = zeros(3, cols);
for index = 1:cols
    gm = kron(kron(U, V), W)' * X_imode(:,index);
    coords(:,index) = gm; %[coords, gm];
end
%visualize g's in 3D
figure
hold on
[x_len, y_len] = size(coords);
scatter3(coords(1, 1:y_len/2), coords(2, 1:y_len/2), coords(2, 1:y_len/2), 'filled', 'g');
scatter3(coords(1, y_len/2 + 1:y_len), coords(2, y_len/2 + 1:y_len), coords(2, y_len/2 + 1:y_len), 'filled','b');
classification_mat = coords';
classification_mat = [classification_mat, z'];

save('sanity_check_Tucker3_output_100.mat')
toc

function vec_image = vectorize_image(img)
    [I,J,K]=size(img);
    vec_image = []; %zeros(I*J,K); % [];
    for k = 1:K
        vec_image = [vec_image; vec(img(:,:,k))];
    end
end

function Y = update(X, U, V, lambda)
    % Fix one of [U V W]
    % Compute so that g as low-D projection
    for m = 1:length(X)
        for l = 1:length(X)

            x_m = X(:, m);
            x_l = X(:, l);

            w_val = w_lm(m, l, lambda);

            Y = kron(V, U);
            Z = Z + w_val * (x_m - x_l) * (x_m - x_l)';

        end
       disp('finished column processing for col '+string(m))
    end
end

function [x1, x2, x3] = unfold(A)
    %unfold tensor A
    [i,j,k] = size(A);
    % x1 = unfold columns by slice
    x1 = zeros(j*k, i);
    for ii=1:i
        x1(:,ii) = reshape(squeeze(A(ii,:,:)),[k*j,1]);
    end
    
    % x2 = turn rows into columns
    x2 = zeros(i*k,j);
    for jj=1:j
        x2(:,jj) = reshape(squeeze(A(:,jj,:)),[i*k,1]);
    end
    
    % x3 = turn fibers into columns
    x3 = zeros(i*j,k);
    for kk=1:k
        x3(:,kk) = reshape(squeeze(A(:,:,kk)),[i*j,1]);
    end
end


function lambda_val = w_lm(z1, z2, lambda)
    coeff = double(z1 < 51 && z2 < 51);
    lambda_val = ((1-lambda)^coeff) * ((-lambda)^(1-coeff)); 
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

%Turn columns of X into vectorized tensors
function X_vec = vectorizeX(X)
    X_vec = [];
    for i = 1:length(X)
        X_vec = [X_vec, vec(X(:,i))];
    end
end

%vectorize tensor slabwise
function vec_image = vectorize_tensor(img)
    [I,J,K]=size(img);
    vec_image = []; %zeros(I*J,K); % [];
    for k = 1:K
        vec_image = [vec_image; vec(img(:,:,k))];
    end
end

function vec_image = vectorize_tensor_imode(img)
    [I,J,K]=size(img);
    vec_image = []; %zeros(I*J,K); % [];
    for i = 1:I
        [a,b,c] = size(img(i,:,:));
        vec_image = [vec_image; reshape(img(i,:,:), [a*b*c, 1])];
    end
end

function vec_image = vectorize_tensor_jmode(img)
    [I,J,K]=size(img);
    vec_image = []; %zeros(I*J,K); % [];
    for j = 1:J
        [a,b,c] = size(img(:,j,:));
        vec_image = [vec_image; reshape(img(:,j,:), [a*b*c, 1])];
    end
end

function vec_image = vectorize_tensor_kmode(img)
    [I,J,K]=size(img);
    vec_image = []; %zeros(I*J,K); % [];
    for k = 1:K
        [a,b,c] = size(img(:,:,k));
        vec_image = [vec_image; reshape(img(:,:,k), [a*b*c, 1])];
    end
end

function [X, z] = buildX(dims, mu, sigma)
    X1 = zeros(dims(1),dims(2),dims(3));
    %X2 = zeros(dims(1),dims(2),dims(3));
    for i = 1:dims(1)
        for j = 1:dims(2)
            for k = 1:dims(3)
                X1(i,j,k) = normrnd(mu, sigma);
                %X2(i,j,k) = normrnd(-mu, sigma);
            end
        end
    end
    X = X1;%[X1, X2];
end

