function [noiX, value] = gauss(X, flag, seed)
% Add IID gauss noise for all view data.
% Input:
%   X: M x N x V tensor, multi-view data
%   flag: 0 or 1, if 0, IID Gausii with varian 0.03; else NIID-Gauss.
%   seed: scalar
% Output:
%   noiX: M x N x V tensor, noise multi-view data
%   value:
%       value.psnr: N x V matrix
%       value.ssim: N x V matrix
%       value.mpsnr: scalar
%       value.mssim: scalar

if nargin < 3
    seed = 101;
end

[M, N, V] = size(X);

rng('default'); rng(seed);
switch flag
    case 0
        variance = 0.02 * ones(1, V);
    case 1
        variance = unifrnd(0.01, 0.05, [1, V]);
end

noiX = X;
for vv = 1:V
    noiX(:, :, vv) = X(:,:,vv) + randn(M, N)*sqrt(variance(vv));
end

imgSize = [128, 96];
value = ssimCal(noiX, imgSize, X);

