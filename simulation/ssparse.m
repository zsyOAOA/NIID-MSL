function [noiX, value] = ssparse(X, flag, seed)
% Add IID laplace distribution sparse noise to clean data,
% Input:
%   X: M x N x V tensor, noise data
%   flag: 0 or 1
%   seed: defautl 101
% Output:
%   noiX: M x N x V tensor, clean data

if nargin == 1
    seed = 101;
end
rng('default'); rng(seed);

[M, N, V] = size(X);
noiRation = 0.1;
switch flag
    case 0
        upUnif = 2*max(abs(X(:))) * ones(1, V); % 1 x V matrix
    case 1
        upUnif = [1, 2, 3, 2, 1] * max(abs(X(:))); % 1 x V matrix
end

downUnif  = -upUnif;

noiX = X;
for jj = 1:V
    for nn = 1:N
        shuffleInd = randperm(M);
        noiInd = shuffleInd(1:floor(M * noiRation));
        noiX(noiInd, nn, jj) = X(noiInd, nn, jj) + unifrnd(downUnif(jj), upUnif(jj), [length(noiInd), 1]);
    end
end

imgSize = [128, 96];
value = ssimCal(noiX, imgSize, X);
