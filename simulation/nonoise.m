function [noiX, value] = nonoise(X)
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

noiX = X;
value = 1;

