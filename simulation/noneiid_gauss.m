function [noiX, value] = noneiid_gauss(X, varDown, varUp, seed)
% Add IID gauss noise for all view data.
% Input:
%   X: M x N x V tensor, multi-view data
%   variance: scalae, default 0.04
%   seed: scalar
% Output:
%   noiX: M x N x V tensor, noise multi-view data
%   value:
%       value.psnr: N x V matrix
%       value.ssim: N x V matrix
%       value.mpsnr: scalar
%       value.mssim: scalar

if nargin == 1
    varUp   = 0.06;
    varDown = 0.02;
    seed    = 101;
end
if nargin == 2
    error('Pleast input paramter varDown and varUp');
end
if nargin == 3
    seed = 101;
end

rng('default'); rng(seed);

[M, N, V] = size(X);
variance = unifrnd(varDown, varUp, V, 1);

noiX = X;
for jj = 1:V
    for nn = 1:N
        noiX(:, nn, jj) = X(:, nn, jj) + randn(M, 1) * sqrt(variance(jj));
    end
end

imgSize = [128, 96];
value = ssimCal(noiX, imgSize, X);
