function value = ssimCal(Noi_X, imgSize, Ori_H)
% Input:
%   Noi_X: M x N x V tensor, noise multi-view data
%   imgSize: image size in each view
%   Ori_H: M x N x V tensor, clean multi-view data
% Output:
%   value: struct
%       value.psnr: N x V matrix
%       value.ssim: N x V matrix

[~, N, V] = size(Noi_X);

psnr = zeros(N, V);
ssim = zeros(N, V);
for jj = 1:V
    Noi_X_V     = reshape(Noi_X(:, :, jj), [imgSize N]);
    Ori_X_V     = reshape(Ori_H(:, :, jj), [imgSize N]);
    valueVV     = zhibiao(Ori_X_V, Noi_X_V);
    psnr(:, jj) = valueVV.psnr;
    ssim(:, jj) = valueVV.ssim;
end
value.psnr = psnr;
value.ssim = ssim;
value.mpsnr = mean(psnr(:));
value.mssim = mean(ssim(:));

