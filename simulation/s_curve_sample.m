clear all
close all

% S-CURVE DATASET
N=4000;
K=12;
d=2;
seed = 101;
rng('default'); rng(seed);


% PLOT TRUE MANIFOLD
tt = [-1:0.1:0.5]*pi; uu = tt(end:-1:1); hh = [0:0.1:1]*5;
xx = [cos(tt) -cos(uu)]'*ones(size(hh));
yy = ones(size([tt uu]))'*hh;
zz = [sin(tt) 2-sin(uu)]'*ones(size(hh));
cc = [tt uu]' * ones(size(hh));

%subplot(1,2,1);
figure(1);
surf(xx,yy,zz,cc);
grid off;
view([12 -20 3]);
%axis([-1,1,0,5,-1,3]);

% GENERATE SAMPLED DATA
angle = pi*(1.5*rand(1,N/2)-1); height = 5*rand(1,N);
X = [[cos(angle), -cos(angle)]; height;[ sin(angle), 2-sin(angle)]];

% SCATTERPLOT OF SAMPLED DATA
figure(2);
scatter3(X(1,:),X(2,:),X(3,:),12,[angle angle]);
grid off;
view([12 -20 3]);

SData = X;
clear X;
XCell = cell(1, 3);
XCell{1} = SData(2:3, :);
XCell{2} = SData([1;3], :);
XCell{3} = SData(1:2, :);
XData = zeros(2, N, 3);
XData(:, :, 1) = SData(2:3, :);
XData(:, :, 2) = SData([1;3], :);
XData(:, :, 3) = SData(1:2, :);
%save('/home/oa/code/matlab/HdpMultiView/data/scurve.mat', 'XData', 'XCell', 'SData', 'angle');

% Projection on X-Y, Y-Z and X-Z plane
sampleShuffle = randperm(N);
ind1 = sampleShuffle(1:1000);
ind0 = sampleShuffle(1:300);
col = [angle angle];
col1 = col(ind1);
col0 = col(ind0);
figure(3);
scatter(SData(1,ind1),SData(2,ind1),20,col1);
figure(4);
scatter(SData(1,ind0),SData(3,ind0),30,col0);
figure(5);
scatter(SData(2,ind1),SData(3,ind1),20,col1);
