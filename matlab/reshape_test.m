clear
Xt = zeros([2,3,5]);
for i = 1:5
    Xt(:,:,i) = i*ones([2,3]);
end
[x1, x2, x3] = unfold(Xt);
disp(x1)
disp(x2)
disp(x3)
v_i = vectorize_tensor_imode(Xt) %vec(x1)
v_j = vectorize_tensor_jmode(Xt) %vec(x2)
v_k = vectorize_tensor_kmode(Xt) %vec(x3)
reshape(v, [2,3,5])

function [x1, x2, x3] = unfold(A)
    %unfold tensor A
    [i,j,k] = size(A);
    % x1 = unfold columns by slice
    x1 = zeros(j*k, i);
    for ii=1:i
        x1(:,ii) = reshape(squeeze(A(ii,:,:)),[j*k,1]);
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