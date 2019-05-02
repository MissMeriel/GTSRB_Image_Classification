clear

figure
load('sanity_check_brev_3Doutput_051.mat')
hold on
scatter3(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), coords(3, 1:dim*sample_sz), 'filled', 'g')
scatter3(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), coords(3, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
title('\lambda = '+string(lambda)+', '+string(sample_sz)+' samples, grayscale image matrices')
disp('3d 0.51 40 samples')
hold off

%figure
%load('sanity_check_3Dscatter_output_050.mat')
%hold on
%scatter3(coords(1, 1:3*sample_sz), coords(2, 1:3*sample_sz), coords(3, 1:3*sample_sz), 'filled', 'g')
%scatter3(coords(1, 3*sample_sz+1:3*sample_sz*2), coords(2, 3*sample_sz+1:3*sample_sz*2), coords(3, 3*sample_sz+1:3*sample_sz*2), 'filled','b')
%title('\lambda = 0.50, 40 samples, RGB slab-vectorized tensor')
%disp('Vectorized RGB slabs of tensor, 40 samples')
%hold off

%figure
%load('sanity_check_scatter_3D_fullvec_output_050.mat')
%hold on
%scatter3(coords(1, 1:sample_sz), coords(2, 1:sample_sz), coords(3, 1:sample_sz), 'filled', 'g')
%scatter3(coords(1, sample_sz+1:sample_sz*2), coords(2, sample_sz+1:sample_sz*2), coords(3, sample_sz+1:sample_sz*2), 'filled','b')
%title('\lambda = 0.50, 40 samples, fully vectorized RGB tensor')
%disp('fully vectorized RGB tensor 40 samples')

figure
load('sanity_check_scatter_3D_fullvec_output_050_55sample.mat')
hold on
scatter3(coords(1, 1:sample_sz), coords(2, 1:sample_sz), coords(3, 1:sample_sz), 'filled', 'g')
scatter3(coords(1, sample_sz+1:sample_sz*2), coords(2, sample_sz+1:sample_sz*2), coords(3, sample_sz+1:sample_sz*2), 'filled','b')
title('\lambda = 0.50, 55 samples, matrix of vectorized RGB tensors')
disp('fully vectorized RGB tensor 55 samples')

figure
load('sanity_check_scatter_3D_fullvec_output_050_120sample.mat')
hold on
scatter3(coords(1, 1:sample_sz), coords(2, 1:sample_sz), coords(3, 1:sample_sz), 'filled', 'g')
scatter3(coords(1, sample_sz+1:sample_sz*2), coords(2, sample_sz+1:sample_sz*2), coords(3, sample_sz+1:sample_sz*2), 'filled','b')
title('\lambda = 0.50, 120 samples, matrix of vectorized RGB tensors')
disp('fully vectorized RGB tensor 120 samples')
