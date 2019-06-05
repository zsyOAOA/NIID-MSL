function [noiX, value] = mixGauss(X, seed)
% Add mixture of Gauss noise to multi-view data, each view is added a mixture gauss noise .
% Input:
%   X: M x N x V tensor, clean data
%   seed: default 101
% Output:
%   noiX: M x N x V tensor, noise data
%   value: struct
%       value.psnr: N x V matrix
%       value.ssim: N x V matrix

if nargin < 2
    seed = 1001;
end
rng('default'); rng(seed);

%val       = [0.01, 0.04, 0.08, 0.2, 0.5];   % Global variance
val       = [0.01, 0.05, 0.1, 0.5, 1];   % Global variance
ratio     = [0.2, 0.3, 0.2, 0.2, 0.1];
localNum = 3;

[M, N, V] = size(X);
noiX = X;
for jj = 1:V
    shuffleInd = randperm(M);
    samLocal = randsample(length(val), localNum);
    valLocal = val(samLocal);
    %ratioLocal = ratio(samLocal) / sum(ratio(samLocal));
    ratioLocal = ratio(samLocal);
    cumRationlocal = cumsum(ratioLocal);
    indCut = floor(M * cumRationlocal);
    for kk = 1:localNum
        if kk == 1
            indStartKK = 1;
        else
            indStartKK = indCut(kk-1);
        end
        indEnd = indCut(kk);
        index = shuffleInd(indStartKK:indEnd)';
        for nn = 1:N
            noiX(index, nn, jj) = X(index, nn, jj) + randn(length(index), 1) * sqrt(valLocal(kk));
        end
    end
end

imgSize = [128, 96];
value = ssimCal(noiX, imgSize, X);

