clear;clc;
%---------------------------------Prepare Data------------------------------------
viewInd = [1;2;3];
V = length(viewInd);
dataPath = './data/background_subtraction/lab.mat';
load(dataPath);  % get cell variable X, X{i}: 144*180 x 200 matrix
imgSize = [144, 180];
imgNum = 200;

oriData = zeros([imgSize(1)*imgSize(2), imgNum, V]);
for jj = 1:V
    oriData(:, :, jj) = X{viewInd(jj)};
end

%-----------------------------Parameter settings----------------------------------
dim       = 3;
k         = 3;
[M, N, ~] = size(oriData);

%----------------------------------Hdp_RMoG---------------------------------------
fprintf ('\nNIID-MSL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n');
opts = set_parameter('initK', 15, 'initT', 5, 'initDim', dim, ...
                        'tol', 1e-4, 'itermax', 30, 'display', 1, 'bound', 1);
[~, lowrank, XHdp] = hdp_multi_view(oriData, opts);
back = mat_times3(lowrank.L, lowrank.R);
fore = oriData - back;

gapx = 10;
gapy = 20;
video_res = ones(imgSize(1)*V+gapy*(V-1), imgSize(2)*3+gapx*2, imgNum);
for ii = 1:V
    for jj = 1:3
        indyStart = imgSize(1)*(ii-1) + gapy*(ii-1) + 1;
        indyEnd = imgSize*ii + gapy*(ii-1);
        indxStart = imgSize(2)*(jj-1) + gapx*(jj-1) + 1;
        indxEnd = imgSize(2)*jj + gapx*(jj-1);
        for kk = 1:imgNum
            switch jj
                case 1
                    video_res(indyStart:indyEnd, indxStart:indxEnd, kk) = std_image(reshape(oriData(:, kk, ii), imgSize));
                case 2
                    video_res(indyStart:indyEnd, indxStart:indxEnd, kk) = std_image(reshape(back(:, kk, ii), imgSize));
                case 3
                    video_res(indyStart:indyEnd, indxStart:indxEnd, kk) = std_image(abs(reshape(fore(:, kk, ii), imgSize)));
            end
        end
    end
end

figure;
for kk = 1:imgNum
    imshow(video_res(:, :, kk));
end

