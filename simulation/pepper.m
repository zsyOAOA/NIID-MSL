function [noiX, value] = pepper(X, densityMax, seed)
% Add 'salt and pepper' block, each picture with two blocks.
% Input:
%   X: M x N x V tensor data, clean data.
%   densityMax: maximum density of pepper noise, default 0.4
%   seed: defatlt 101
% Output:
%   noiX： M x N x V tensor, noise data.

if nargin == 1
    densityMax = 0.8;
    seed       = 101;
end
if nargin == 2
    seed = 101;
end
rng('default'); rng(seed);

height     = 128;
width      = 96;
[~, N, V]  = size(X);
density    = unifrnd(0.5, densityMax, [V, N]);
blockNum = 2;

noiX = X;
for jj = 1:V
    for nn = 1:N
        for kk = 1:blockNum
            img                    = reshape(noiX(:, nn, jj), [height, width]);
            imgNoi                 = img;
            blockHeight            = randsample(15:40, 1);
            blockWidth             = randsample(15:30, 1);
            heightStart            = randsample(height-blockHeight, 1);
            widthStart             = randsample(width - blockWidth, 1);
            indexH                 = heightStart:(heightStart+blockHeight);
            indexW                 = widthStart:(widthStart+blockWidth);
            imgNoi(indexH, indexW) = imnoise(img(indexH, indexW), 'salt & pepper', density(jj, nn));
            noiX(:, nn, jj)        = imgNoi(:);
        end
    end
end

value = ssimCal(noiX, [height, width], X);
