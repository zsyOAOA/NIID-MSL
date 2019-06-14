function X = Dcmall2MulData()
DcmallPath = '/home/oa/code/matlab/Data/indian_std.mat';
load(DcmallPath);  % get Ori_H

V = 4;
[H, W, B] = size(Ori_H);
M = H * W;
N = floor(B / V);

X = zeros(M, N, V);
for jj = 1:V
    X(:,:,jj) = reshape(Ori_H(:, :, N*(jj-1)+1:N*jj), [M, N]);
end
