clear;
clc;
seed = 10001;
rng('default'); rng(seed);

data_key = 'Campus';
initK_hdp = 25; initT_hdp = 15; r = 3; iter_hdp = 15;

%data_key = 'Escalator';
%initK_hdp = 3; initT_hdp = 1; r = 1; iter_hdp = 15;

%data_key = 'Fountain';  % ground_ind = 1,2,3,4
%initK_hdp = 15; initT_hdp = 5; r = 3; iter_hdp = 15;

%data_key = 'Curtain';
%initK_hdp = 25; initT_hdp = 15; r = 1; iter_hdp = 20;

%data_key = 'Hall';
%initK_hdp = 25; initT_hdp = 15; r = 3; iter_hdp = 15;

%data_key = 'Lobby';
%initK_hdp = 25; initT_hdp = 15; r = 3; iter_hdp = 15;

%data_key = 'ShoppingMall';
%initK_hdp = 3; initT_hdp = 1; r = 1; iter_hdp = 15;

%data_key = 'WaterSurface';
%initK_hdp = 15; initT_hdp = 5; r = 1; iter_hdp = 15;

data_path = fullfile('./data/foreground_detection/I2R', [data_key, '.mat']);
load(data_path);   % get variable: X_video: h x w x 3 x n tensor
                   %          X_multi_view: h*w x n x 3 tensor
                   %           groundtruth: h*w x 1 vector
                   %       ind_groundtruth: integer, index of groundtruth
[h, w, ~, ~] = size(X_video);
[D, N, V] = size(X_multi_view);

X_multi_view_cell = cell(V, 1);  % D x N for each cell
for ii = 1:V
    X_multi_view_cell{ii} = X_multi_view(:, :, ii);
end

fprintf ('\nNIID-MSL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n');
opt_hdp_cell = set_parameter('initK', initK_hdp, 'initT', initT_hdp, 'init', 2, 'initDim', r,...
                                            'tol', 1e-4, 'itermax', iter_hdp, 'bound', 0, 'cutK', 1, 'mergeT', 1);
[~, ~, XHdp_cell] = hdp_multi_view_cell(X_multi_view_cell, opt_hdp_cell);
XHdp_cell2mat = reshape(cell2mat(XHdp_cell), [h, w, V, N]);
err_video_hdp_cell = abs(permute(reshape(X_multi_view, [h, w, N, V]), [1,2,4,3])- XHdp_cell2mat);

[F_measure_val, thres_best, ~, ~] = F_measure(err_video_hdp_cell(:, :, :, ind_groundtruth), groundtruth);
fprintf('%s video: F-Measure = %.4f...\n', data_key, F_measure_val);

