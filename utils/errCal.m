function [RRSE, RRAE] = errCal(Noi_X, imgSize, Ori_H)
% Input:
%   Noi_X: M x N x V tensor, noise multi-view data
%   imgSize: image size in each view
%   Ori_H: M x N x V tensor, clean multi-view data
% Output:
%   RRSE: relative reconstruction square error,
%         (V+1) x 1 vector, corresponding each view and mean
%   RRAE: relative reconstruction abosuluteerror,
%         (V+1) x 1 vector, corresponding each view and mean

[~, N, V] = size(Noi_X);

RRAE = zeros(V+1, 1);
RRSE = zeros(V+1, 1);

for jj = 1:V
    noiseJj  = Noi_X(:, :, jj);
    cleanJj  = Ori_H(:, :, jj);
    RRAE(jj) = sum(abs(noiseJj(:) - cleanJj(:))) / sum(abs(cleanJj(:)));
    RRSE(jj) = norm(noiseJj(:)-cleanJj(:), 2)^2 / norm(cleanJj(:), 2)^2;
end

RRAE(V+1) = sum(abs(Noi_X(:)-Ori_H(:))) / sum(abs(Ori_H(:)));
RRSE(V+1) = norm(Noi_X(:)-Ori_H(:), 2)^2 / norm(Ori_H(:), 2)^2;
end

