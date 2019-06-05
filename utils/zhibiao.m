function  value = zhibiao(Ori_H,Denoi_HSI)
% Input:
%   Ori_H: M x N x B tensor, clean data
%   Denoi_HSI: M x N x B tensor, recovered data
% Outut:
%   value: struct
%       value.psnr: B x 1 vector
%       value.mpsnr: scalar
%       value.ssim: B x 1 vector
%       value.mssim: scalar


[M,N,B] = size(Ori_H);
T1      = reshape(Ori_H*255,M*N,B);
T2      = reshape(Denoi_HSI*255,M*N,B);
temp    = reshape(sum((T1 -T2).^2),B,1)/(M*N); % B x 1 vector
PSNR    = 20*log10(255)-10*log10(temp); % B x 1 vector
MPSNR   = mean(PSNR);
for i=1:B
    [mssim, ssim_map] = ssim_index(Ori_H(:,:,i)*255, Denoi_HSI(:,:,i)*255);
    SSIM(i)           = mssim;
end
MSSIM       = mean(SSIM);
value.psnr  = PSNR;
value.mpsnr = MPSNR;
value.ssim  = SSIM;
value.mssim = MSSIM;
