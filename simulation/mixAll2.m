function [noiX, value] = mixAll2(X, seed)
%Add noise to clean multi-view data, each view is randomly added two kind of the following noise:
%       1: Gauss, variance o.01
%       2: uniform sparse: [-unifTime*max(data), unifTime*max(data)]
%       3: block: densityMax, default 0.4

if nargin == 1
    seed = 101;
end
rng('default'); rng(seed);


[M, N, V]   = size(X);
variance    = 0.02;
unifTime    = 3;
ratioSpasre = 0.1;
densityMax  = 0.3;
height      = 128;
width       = 96;

%noiseType = zeros(V, 2);
%for jj = 1:V
    %noiseType(jj, :) = randsample(3, 2, false);
%end
noiseTypeInit = [1, 3;
                 1, 2;
                 2, 3;
                 1, 2;
                 2, 3];
noiseType = noiseTypeInit(1:V, :);


noiX = X;
for jj = 1:V
    for tt = 1:length(unique(noiseType(jj,:)))
        noiseInd = noiseType(jj, tt);
        switch noiseInd
            %add Gauss noise
            case 1
                for nn = 1:N
                    noiX(:, nn, jj) = noiX(:, nn, jj) + randn(M, 1) * sqrt(variance);
                end
            %add sparse noise
            case 2
                for nn = 1:N
                    randIndex = randperm(M);
                    indexSparse = randIndex(1:floor(M*ratioSpasre));
                    maxNnJj = max(X(:, nn, jj));
                    noiX(indexSparse, nn, jj) = noiX(indexSparse, nn, jj) + ...
                         unifrnd(-unifTime*maxNnJj, unifTime*maxNnJj, [floor(M*ratioSpasre), 1]);
                end
            %add block noise
            case 3
                for nn = 1:N
                    for ss = 1:2
                        img                    = reshape(noiX(:, nn, jj), [height, width]);
                        imgNoi                 = img;
                        blockHeight            = randsample(10:50, 1);
                        blockWidth             = randsample(10:40, 1);
                        heightStart            = randsample(height-blockHeight, 1);
                        widthStart             = randsample(width - blockWidth, 1);
                        indexH                 = heightStart:(heightStart+blockHeight);
                        indexW                 = widthStart:(widthStart+blockWidth);
                        density                = rand(1) * densityMax + 0.4;
                        imgNoi(indexH, indexW) = imnoise(img(indexH, indexW), 'salt & pepper', density);
                        noiX(:, nn, jj)        = imgNoi(:);
                    end
                end
        end
    end
end

value = ssimCal(noiX, [height, width], X);

