function [noiX, value] = mixAll1(X, seed)
%Add noise to clean multi-view data, each view is randomly added one kind of the following noise:
%       1: Gauss, variance o.01
%       2: uniform sparse: [-unifTime*max(data), unifTime*max(data)]
%       3: block: densityMax, default 0.4

if nargin == 1
    seed = 101
end

variance    = 0.01;
unifTime    = 2;
ratioSpasre = 0.1;
densityMax  = 0.4;
height      = 128;
width       = 96;

[M, N, V] = size(X);
noiseType = randsample(3, V, ture);

noiX = X;
for jj = 1:V
    noiseInd = noiseType(jj);
    switch noiseInd
        %add Gauss noise
        case 1
            for nn = 1:N
                noiX(:, nn, jj) = X(:, nn, jj) + randn(M, 1) * sqrt(variance);
            end
        %add uniform noise
        case 2
            for nn = 1:N
                randIndex = randperm(M);
                indexSparse = randIndex(1:floor(M*ratioSpasre));
                maxNnJj = max(X(:, nn, jj));
                noiX(indexSparse, nn, jj) = X(indexSparse, nn, jj) + ...
                          unifrnd(-unifTime*maxNnJj, unifrnd*maxNnJj, [floor(M*ratioSpasre), 1]);
            end
        %add block noise
        case 3
            for nn = 1:N
                img                    = reshape(X(:, nn, jj), [height, width]);
                imgNoi                 = img;
                blockHeight            = randsample(10:50, 1);
                blockWidth             = randsample(10:40, 1);
                heightStart            = randsample(height-blockHeight, 1);
                widthStart             = randsample(width - blockWidth, 1);
                indexH                 = heightStart:(heightStart+blockHeight);
                indexW                 = widthStart:(widthStart+blockWidth);
                density                = rand(1) * densityMax + 0.01;
                imgNoi(indexH, indexW) = imnoise(img(indexH, indexW), 'salt & pepper', density);
                noiX(:, nn, jj)        = imgNoi(:);
            end
    end
end

value = ssimCal(noiX, [height, width], X);

