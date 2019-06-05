clear;
clc
close all;
dataPath = './data/scurve.mat';
load(dataPath); %get XData, XCell, SData, angle; XCell correponds 's' in 2-dimension plane

%parameter settings
V         = 3;
dim       = 3;
[M, N, ~] = size(XData); % XData: 2 x 4000 x 3 Data
initK     = 25;
initT     = 15;

% SNR settings
seed = 10001;
rng('default'); rng(seed);

figure('Name', 'S-curve Experiment');
for ii = 1:6
    switch ii
        case 1
        case 2
            SNR = [20; 20; 20];
        case 3
            SNR = [15; 20; 20];
        case 4
            SNR = [15; 20; 25];
        case 5
            SNR = [10; 20; 20];
        case 6
            %SNR = [10; 20; 30];
            SNR = [30; 20; 10]; % best
    end

    noiseData = XData;
    if ii > 1
        sigma = zeros(V,1);
        for jj = 1:V
            datajj = reshape(XData(:, :, jj), [], 1);
            mu = mean(datajj);
            sigmaDatajj = sum((datajj-mu).^2)/length(datajj);
            sigma(jj) = sigmaDatajj / (10^(SNR(jj)/10));
            noiseData(:, :, jj) = XData(:, :, jj) + randn(M, N) * sqrt(sigma(jj));
        end
    end

    %----------------------------------NIID-MSL---------------------------------------
    itermax = 30;
    opts = set_parameter('initK', initK, 'initT', initT, 'initDim', dim, ...
                            'display', 0, 'tol', 1e-6, 'init', 1, 'itermax', itermax, 'bound', 1);
    [~, ~, XHdp] = hdp_multi_view(noiseData, opts);
    SHdp = [XHdp(:, :, 1); XHdp(2, :, 2)];
    rrse = norm(SHdp(:)-SData(:), 2) / norm(SData(:), 2);
    rrae = sum(abs(SHdp(:)-SData(:))) / sum(abs(SData(:)));
    fprintf('Case: %d, RRSE = %.4e, RRAE = %.4e...\n', ii, rrse, rrae);

    %Figure
    subplot(2, 3, ii);
    scatter3(SHdp(1, :),SHdp(2, :), SHdp(3, :), 12, [angle angle]);
    title(['Case:', num2str(ii)]);
    axis([-1, 1, 0, 5, -1, 3]);
    grid off;
    view([12 -20 3]);
end


