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
datasetRootDir = 'GTSRB_subset/Final_Training/Images/00003/'; %uigetdir; %gets current directory
imagesArr = dir(fullfile(datasetRootDir,'*.ppm')); %gets all .ppm files in struct
X = zeros(30, 30, length(imagesArr));
for k = 1:length(imagesArr)
    filename = strcat(datasetRootDir, imagesArr(k).name);
    temp = imread(filename);
    temp = rgb2gray(temp);
    [x, y] = size(temp);
    if((30 <= x) && (x <= 35) && (30 <= y) && (y <= 35))
        X(:,:,k) = temp(1:30, 1:30);
    else
        disp(strcat("Deleting ",filename));
        delete(filename);
    end
    %X(:,:,k) = temp;
end
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
[i, j, k] = size(X);
compression_ratio = (i*j*k)/(i*r1 + j*r2 + k*r3 + r1*r2*r3)

%the three different matrix unfolding SVDs
%x1
[Ux1,Sx1,Vx1]=svds(x1,2);
SVDerror1=norm(x1-Ux1*Sx1*Vx1','fro')^2;
rel_svd_error_x1 = SVDerror1/signal2
%x2
[Ux2,Sx2,Vx2]=svds(x2,2);
SVDerror2=norm(x2-Ux2*Sx2*Vx2','fro')^2;
rel_svd_error_x2 = SVDerror2/signal2
%x3
[Ux3,Sx3,Vx3]=svds(x3,4);
SVDerror3=norm(x3-Ux3*Sx3*Vx3','fro')^2;
rel_svd_error_x3_under = SVDerror3/signal2
[Ux3,Sx3,Vx3]=svds(x3,5);
SVDerror3=norm(x3-Ux3*Sx3*Vx3','fro')^2;
rel_svd_error_x3_over = SVDerror3/signal2

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