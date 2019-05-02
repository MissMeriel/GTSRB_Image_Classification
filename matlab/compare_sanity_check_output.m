clear
figure

load('sanity_check_brev_3Doutput_051.mat')
hold on
%scatter3(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), coords(3, 1:dim*sample_sz), 'filled', 'g')
%scatter3(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), coords(3, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
disp('3d 0.51 40 samples')
hold off

load('sanity_check_brev_3Doutput_051_55sample.mat')
hold on
scatter3(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), coords(3, 1:dim*sample_sz), 'filled', 'g')
scatter3(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), coords(3, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
disp('3d 0.51 55 samples')

figure
load('sanity_check_brev_output_000.mat')
%subplot(3,2,1);
hold on
V_min = V(:,D_len-2:D_len);
mins = min(D);
if(mins(1) < 0)
    V_min = V(:,1:3);
end
coords = V_min' * X;
scatter3(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), coords(3, 1:dim*sample_sz), 'filled', 'g')
scatter3(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), coords(3, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
title('\lambda = 0.0')
disp('3d 0.00 '+string(sample_sz)+' samples')

clear
load('sanity_check_brev_output_00000001.mat')
%subplot(3,2,2);
figure
hold on
V_min = V(:,D_len-2:D_len);
mins = min(D);
if(mins(1) < 0)
    V_min = V(:,1:3);
end
coords = V_min' * X;
[i,j]=size(coords)
scatter3(coords(1, 1:j/2), coords(2, 1:j/2), coords(3, 1:j/2), 'filled', 'g')
scatter3(coords(1, j/2+1:j), coords(2, j/2+1:j),coords(3, j/2+1:j), 'filled','b')
title('\lambda = 0.0000001')
disp('3d 0.0000001 '+string(sample_sz)+' samples')

%load('sanity_check_brev_output_00001.mat')
%subplot(3,2,2);
%hold on
%scatter(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), 'filled', 'g')
%scatter(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
%title('\lambda = 0.0001')

load('sanity_check_brev_output_025.mat')
%subplot(3,2,3);
figure
hold on
V_min = V(:,D_len-2:D_len);
mins = min(D);
if(mins(1) < 0)
    V_min = V(:,1:3);
end
coords = V_min' * X;
scatter3(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), coords(3, 1:dim*sample_sz), 'filled', 'g')
scatter3(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), coords(3, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
%scatter(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), 'filled', 'g')
%scatter(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
title('\lambda = '+string(lambda)+', '+string(sample_sz)+' samples')
disp('3d 0.25 '+string(sample_sz)+' samples')

load('sanity_check_brev_output_050.mat')
subplot(3,2,4);
hold on
scatter(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), 'filled', 'g')
scatter(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
title('\lambda = '+string(lambda)+', '+string(sample_sz)+' samples')

%load('sanity_check_brev_output_055.mat')
%load('sanity_check_brev_output_060.mat')
%load('sanity_check_brev_output_065.mat')
%load('sanity_check_brev_output_070.mat')

load('sanity_check_brev_output_075.mat')
subplot(3,2,5);
hold on
scatter(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), 'filled', 'g')
scatter(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
title('\lambda = 0.75')

load('sanity_check_brev_output_100.mat')
subplot(3,2,6);
hold on
scatter(coords(1, 1:dim*sample_sz), coords(2, 1:dim*sample_sz), 'filled', 'g')
scatter(coords(1, dim*sample_sz+1:dim*sample_sz*2), coords(2, dim*sample_sz+1:dim*sample_sz*2), 'filled','b')
title('\lambda = 1.00')

%hold on
%scatter(coords(1, 1:1500), coords(2, 1:1500), 'filled', 'g')
%scatter(coords(1, 1501:3000), coords(2, 1501:3000), 'filled','b')