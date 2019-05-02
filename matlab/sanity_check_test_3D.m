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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

%create test tensor
dim = 50;
X = buildX(dim, 1, 0.1)
X = vectorizeX(X);

disp('finished matrix concatenation')

r1 = 1;
r2 = 1;
r3 = 1;
[x1, x2, x3] = unfold(X);
[U1, S1, V1] = svd(x1);
[U2, S2, V2] = svd(x2);
[U3, S3, V3] = svd(x3);
U = V1(:,1:r1);
V = V2(:,1:r2);
W = V3(:,1:r3);


% _|_Tucker3 test
signal2=0;
for kk=1:k
    signal2=signal2+norm(X(:,:,kk),'fro')^2;
end


lambda = 0.0001;
Y = zeros(size(X1));


n = 0;
[i,j,k]=size(X);
rel_tucker_errors = [];
oldEstimateNorm = 1;
estimateNorm = 0;
while abs(oldEstimateNorm - estimateNorm) > 0.9
    n = n + 1;
    %compute new U V W
    [uu1, us1, uv1] = svd(kron(W,V)' * x1);
    [vu1, vs1, vv1] = svd(kron(W,U)' * x2);
    [wu1, ws1, wv1] = svd(kron(V,U)' * x3);
    
    %update U V W
    U = uv1(:,1:r1);
    V = vv1(:,1:r2);
    W = wv1(:,1:r3);
    
    %calc G
    G3 = kron(V,U)' * x3 * W;
    G = reshape(G3, [r1 r2 r3]);

    %calc Xm (aka X~)
    X3 = kron(V,U) * G3 * W';
    Xm = reshape(X3, size(X));

    %calc error & convergence
    error2=0;
    oldEstimateNorm = estimateNorm;
    estimateNorm = 0;
    for kk=1:k
        error2=error2+norm(X(:,:,kk)-Xm(:,:,kk),'fro')^2;
        estimateNorm = estimateNorm+norm(Xm(:,:,kk),'fro')^2;
    end
    rel_tucker_error = error2/signal2;
    rel_tucker_errors(n) =  rel_tucker_error;
    
    %plot relative tucker error
    pl = plot(rel_tucker_errors, 'g');
    title('\fontsize{24}GTSRB Data Subset Decomposition');
    xlabel('\fontsize{16}Steps')
    ylabel('\fontsize{16}Relative Tucker error')
    pause(0);
end

% compute g's

whos Y
[V, D] = eigs(Y)
D_len = size(D);
D_len = D_len(2);
V_min = V(:,D_len-1:D_len);
coords = V_min' * X;

hold on
scatter(coords(1, 1:50), coords(2, 1:50), 'filled', 'g')
scatter(coords(1, 51:100), coords(2, 51:100), 'filled','b')
save('sanity_check_3D_test_output_00001.mat')


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
    for i = 1:length(X)
        X_vec = [X_vec, vec(X())];
    end
end

function X = buildX(dim, mu, sigma)
    X1 = zeros(dim,dim,dim);
    X2 = zeros(dim,dim,dim);
    for i = 1:length(X1)
        for j = 1:length(X1)
            for k = 1:length(X1)
                X1(i,j,k) = normrnd(mu, sigma);
                X2(i,j,k) = normrnd(-mu, sigma);
            end
        end
    end
    X = [X1, X2];
end