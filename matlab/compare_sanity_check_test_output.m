clear
load('sanity_check_test_output_000.mat')
figure

subplot(3,2,1);
hold on
scatter(coords(1, 1:dim), coords(2, 1:dim), 'filled', 'g')
scatter(coords(1, dim+1:dim*2), coords(2, dim+1:dim*2), 'filled','b')
title('\lambda = 0.0')

load('sanity_check_test_output_000001.mat')
subplot(3,2,2);
hold on
scatter(coords(1, 1:dim), coords(2, 1:dim), 'filled', 'g')
scatter(coords(1, dim+1:dim*2), coords(2, dim+1:dim*2), 'filled','b')
title('\lambda = 0.00001')

load('sanity_check_test_output_025.mat')
%load('sanity_check_test_output_00001.mat')
subplot(3,2,3);
hold on
scatter(coords(1, 1:dim), coords(2, 1:dim), 'filled', 'g')
scatter(coords(1, dim+1:dim*2), coords(2, dim+1:dim*2), 'filled','b')
title('\lambda = 0.25')
%title('\lambda = 0.0001')

load('sanity_check_test_output_050.mat')
%load('sanity_check_test_output_0001.mat')
subplot(3,2,4);
hold on
scatter(coords(1, 1:dim), coords(2, 1:dim), 'filled', 'g')
scatter(coords(1, dim+1:dim*2), coords(2, dim+1:dim*2), 'filled','b')
title('\lambda = 0.50')
%title('\lambda = 0.001')

load('sanity_check_brev_output_055.mat')
load('sanity_check_brev_output_060.mat')
load('sanity_check_brev_output_065.mat')
load('sanity_check_brev_output_070.mat')

load('sanity_check_test_output_075.mat')
%load('sanity_check_test_output_001.mat')
subplot(3,2,5);
hold on
scatter(coords(1, 1:dim), coords(2, 1:dim), 'filled', 'g')
scatter(coords(1, dim+1:dim*2), coords(2, dim+1:dim*2), 'filled','b')
title('\lambda = 0.75')
%title('\lambda = 0.01')

load('sanity_check_test_output_100.mat')
subplot(3,2,6);
hold on
scatter(coords(1, 1:dim), coords(2, 1:dim), 'filled', 'g')
scatter(coords(1, dim+1:dim*2), coords(2, dim+1:dim*2), 'filled','b')
title('\lambda = 1.00')

%hold on
%scatter(coords(1, 1:1500), coords(2, 1:1500), 'filled', 'g')
%scatter(coords(1, 1501:3000), coords(2, 1501:3000), 'filled','b')