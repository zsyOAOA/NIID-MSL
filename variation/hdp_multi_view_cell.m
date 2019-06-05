function [mogParam, lowRankRes, recover] = hdp_multi_view_cell(noiseData, opts)
%Input:
%   noiseData: V x 1 cell, cell{v}: D x N matrix data, N is number of data samples, D is the dimensionality
%   opts: struct, returned by function set_parameter.
%Output:
%   mogParam: MoG parameter, struct.
%       mogParam.rau: V x 1 cell, cell{v}: D x N x T tensor, responsibility matrix of local MoG
%       mogParam.pphi: V x 1 cell, cell{v}: T x K tensor, responsibility matrix of Global MoG.
%   lowRankRes: Low Rank parameter. struct.
%       lowRankRes.L: V x 1 cell, cell{v}: D x r matrix
%       lowRankRes.R: r x N matrix
%   recover:  V x 1 cell, cell{v}: D x N matrix

V = length(noiseData);
opts.N = size(noiseData{1}, 2);
opts.DCell = cellfun(@(x) size(x,1), noiseData, 'UniformOutput', false);
opts.TIter = mat2cell(ones(V, 1)*opts.initT, ones(V, 1));

%Initialization
[mogParam, lowRankRes, exErrSquare, sigma, opts] = hdp_init(noiseData, opts);

iter = 1;
currentTol = Inf;
bound = -Inf(opts.itermax, 1);
exLogRes = struct();
while (iter <= opts.itermax)
    opts.iter = iter;
    [mogParam, exLogRes, opts] = hdp_maxv(mogParam, opts, exErrSquare, exLogRes);
    [mogParam, lowRankRes, exErrSquare, exLogRes, sigma] = hdp_low_rank(noiseData,...
                                                mogParam, lowRankRes, opts, exLogRes, sigma);
    [mogParam, exLogGauss, exLogRes, opts] = hdp_exp(mogParam, exErrSquare, exLogRes, opts);
    if opts.display
        if opts.bound
            bound(iter) = hdp_low_bound(mogParam, lowRankRes, opts, exLogGauss, exLogRes, sigma);
            if iter > 1
                currentTol = abs(bound(iter)-bound(iter-1)) / abs(bound(iter-1));
            end
            fprintf('Iter=%2d, bound=%.8f, tol=%.6f...\n', iter, bound(iter), currentTol);
            if (currentTol < opts.tol)
                break;
            end
        else
            fprintf('Iter=%2d...\n', iter);
        end
    end
    iter = iter + 1;
end
mogParam.bound = bound(1:iter-1);
recover = cellfun(@(x) x*lowRankRes.R, lowRankRes.L, ...
                                                'UniformOutput', false);  % V x 1 cell, D x N
end

function [mogParam, lowRankRes, exErrSquare, sigma, opts] = hdp_init(noiseData, opts)
%Output:
%   sigma:
%       sigma.L: V x 1 cell, cell{v}: r x r  x D tensor
%       sigma.R: r x r x N
%   exErrSquare: V x 1 cell, cell{v}: D x N

r = opts.initDim;
D = opts.DCell;
N = opts.N;
V = length(noiseData);

% Initialize L ran R
noiseResize = cell2mat(noiseData);
[LAll, R0, ~,~] = PCA(noiseResize, r);
R = R0';  % R: r x N
L = mat2cell(LAll, cell2mat(D));  % V x 1 cell
lowRankRes.L      = L;
lowRankRes.R      = R;
if opts.display
    disp('PCA Initialization.');
end

scale = cellfun(@data_scale, noiseData, 'UniformOutput', false);  % V x 1 cell
sigma.L = cellfun(@(s,d) repmat(eye(r)*s, [1,1,d]), scale, D,...
                                                   'UniformOutput', false);  % V x 1 cell
sigma.R = repmat(eye(r)*mean(cell2mat(scale)), [1, 1, N]);  % r x r x N

XReconstruct = cellfun(@(x) x*R, L, 'UniformOutput', false); % V x 1 cell tensor
err = cellfun(@(x,y) x-y, noiseData, XReconstruct, 'UniformOutput', false);  % V x 1 cell
errVecCell = cellfun(@(x) reshape(x, [], 1), err, 'UniformOutput', false); % V x 1 cell
exErrSquare = cellfun(@(x) x.^2, err, 'UniformOutput', false); % V x 1 cell

mogParam.pphi = cell(V, 1);
mogParam.rau = cell(V, 1);

% Method 1: Initialize rau ang phi
K = opts.initK;
errVec = cell2mat(errVecCell);
num_K_init = min(K*50, length(errVec));
KCenter_init = datasample(errVec, num_K_init, 'Replace', false);
KCenter = kmeans_dim1(KCenter_init, K, 5);
%KCenter = datasample(errVec, K*50, 'Replace', false);
indStart = 1;
indEnd = 0;
for jj = 1:V
    indEnd = indEnd + D{jj};
    if jj > 1
        indStart = indStart + D{jj-1};
    end
    indT = randsample(K, opts.initT);
    TCenter = KCenter(indT);
    dist = bsxfun(@minus, errVecCell{jj}, TCenter').^2;
    [~, label] = min(dist, [], 2); % D*N x 1
    [labelU, ~, ic]= unique(label);
    Tjj = length(labelU);
    opts.TIter{jj} = Tjj;
    rau0_jj = full(sparse(1:D{jj}*N, ic, 1, D{jj}*N, Tjj, D{jj}*N));  % D*N x T
    mogParam.rau{jj} = reshape(rau0_jj, D{jj}, N, Tjj);
    mogParam.pphi{jj} = full(sparse(1:Tjj, indT(labelU), 1, Tjj, K, Tjj));
end

% Method 3: Initialize rau and phi
%K = opts.initK;
%initT = opts.initT;
%KCenter = zeros(V*initT, 1);
%for jj = 1:V
    %tic;
    %[centerjj, label_jj] = kmeans_dim1(errVecCell{jj}, initT, 10);
    %toc;
    %KCenter((jj-1)*initT+1:jj*initT) = centerjj;
    %rau_jj = full(sparse(1:D{jj}*N, label_jj, 1, D{jj}*N, initT, D{jj}*N));
    %mogParam.rau{jj} = reshape(rau_jj, D{jj}, N, initT);
%end
%[~, labelTK] = kmeans_dim1(KCenter, K, 50);
%for jj = 1:V
    %temp = labelTK((jj-1)*initT+1:jj*initT);
    %mogParam.pphi{jj} = full(sparse(1:initT, temp, 1, initT, K, initT));
%end
end

function scale = data_scale(data)
% data: D x N matrix
[D, N] = size(data);
scale = sum(data(:).^2) / (D*N);
scale = sqrt(scale);
end

function [mogParam, exLogRes, opts] = hdp_maxv(mogParam, opts, exErrSquare, exLogRes)
%   exLogRes:
%       exLogRes.LogPiPie: E[log({\Pi_t^v}^{'})], V x 1 cell
%       exLogRes.LogPiPie1: E[1-log({\Pi_t^v}^{'})], V x 1 cell
%       exLogRes.LogPi: E[log(\Pi_t^v)], V x 1 cell
%       exLogRes.LogBetaPie: E[log({\beta_k}^{'})]
%       exLogRes.LogBetaPie1: E[1-log({\beta_k}^{'})]
%       exLogRes.LogXi: E[log(\xi_k)]
%       exLogRes.LogLambda: E[log(\lambda)]
%       exLogRes.LogGamma: E{log(\gamma)}
%       exLogRes.LogAlpha: E[log(\alpha)]

N = opts.N;
D = opts.DCell;   % V x 1 cell
V = length(exErrSquare);
K = opts.initK;

% merge T componets for each view
if opts.mergeT && opts.iter > 1
    for jj = 1:V
        ind_merge = merge_T(mogParam.pphi{jj}, opts.merge_thres);
        opts.TIter{jj} = length(ind_merge);
        rau_new = zeros(D{jj}, N, opts.TIter{jj});
        pphi_new = zeros(opts.TIter{jj}, K);
        for kk = 1:length(ind_merge)
            rau_new(:, :, kk) = sum(mogParam.rau{jj}(:, :, ind_merge{kk}), 3);
            pphi_new(kk, :) = sum(mogParam.pphi{jj}(ind_merge{kk}, :), 1);
        end
        pphi_new = bsxfun(@rdivide, pphi_new, sum(pphi_new, 2));
        mogParam.rau{jj} = rau_new;
        mogParam.pphi{jj} = pphi_new;
    end
end
T = opts.TIter;

%Update q(\pi');
if ~isfield(mogParam, 'alpha')
    mogParam.alpha = opts.m0 /opts.n0 * ones(V, 1);
end
mogParam.rrPi1 = cell(V, 1);
mogParam.rrPi2 = cell(V, 1);
exLogRes.LogPiPie = cell(V, 1);
exLogRes.LogPiPie1 = cell(V, 1);
exLogRes.LogPi = cell(V, 1);
for jj = 1:V
    numVT = sum(reshape(mogParam.rau{jj}, [], T{jj}), 1); % 1 x T
    rrPi1 = numVT + 1;
    rrPi2 = bsxfun(@minus, sum(numVT), cumsum(numVT)) + mogParam.alpha(jj); % 1 x T
    mogParam.rrPi1{jj} = rrPi1;
    mogParam.rrPi2{jj} = rrPi2;
    temp = psi(rrPi1 + rrPi2);
    exLogRes.LogPiPie{jj}  = psi(rrPi1) - temp; % 1 x T
    exLogRes.LogPiPie1{jj} = psi(rrPi2) - temp; % 1 x T
    cumExLogPiPie1 = [0, cumsum(exLogRes.LogPiPie1{jj}(1:(end-1)))];
    exLogRes.LogPi{jj} = exLogRes.LogPiPie{jj} + cumExLogPiPie1;
    exLogRes.LogPi{jj} = exLogRes.LogPi{jj} / sum(exLogRes.LogPi{jj});
end

%Update q(\alpha)
mm = cell2mat(T) + opts.m0; % V x 1
nn = opts.n0 - cellfun(@sum, exLogRes.LogPiPie1); % V vector
mogParam.alpha = mm ./ nn;
mogParam.mm = mm;
mogParam.nn = nn;
exLogRes.LogAlpha = psi(mm) - reallog(nn);

%Update q(beta')
if ~isfield(mogParam, 'ggamma')
    mogParam.ggamma = opts.g0 / opts.h0;
end
ssBeta1 = ones(K, 1);
for jj = 1:V
    ssBeta1 = ssBeta1 + reshape(sum(mogParam.pphi{jj}, 1), [], 1);
    ssBeta2 = sum(ssBeta1) - cumsum(ssBeta1) + mogParam.ggamma;
end
temp = psi(ssBeta1 + ssBeta2);
mogParam.ssBeta1 = ssBeta1;
mogParam.ssBeta2 = ssBeta2;
exLogRes.LogBetaPie  = psi(ssBeta1) - temp; % K vector
exLogRes.LogBetaPie1 = psi(ssBeta2) - temp; % K vector
cumExLogBeatPie1 = [0; cumsum(exLogRes.LogBetaPie1(1:(end-1)))];
exLogRes.LogBeta = exLogRes.LogBetaPie + cumExLogBeatPie1;
exLogRes.LogBeta = exLogRes.LogBeta / sum(exLogRes.LogBeta);
clear cumExLogBeatPie1;

%Update q(\gamma)
gg              = K + opts.g0;
hh              = opts.h0 - sum(exLogRes.LogBetaPie1);
mogParam.ggamma = gg / hh;
exLogRes.LogGamma = psi(gg) - reallog(hh);
mogParam.gg = gg;
mogParam.hh = hh;

%Update q(\xi)
ee = ones(K, 1) * opts.e0;
ff = ones(K, 1) * opts.f0;
for jj = 1:V
    phiResize = mogParam.pphi{jj}; %T x K matrix
    numDataK = phiResize' * (reshape(mogParam.rau{jj}, [D{jj}*N, T{jj}])'*ones(D{jj}*N, 1)); %K x 1
    ee = ee + 0.5*numDataK; %K x 1 vector
    rauErrSquare = bsxfun(@times, mogParam.rau{jj}, exErrSquare{jj}); % D x N x T tensor
    rauPhiErrSquare = phiResize' * (reshape(rauErrSquare, [N*D{jj}, T{jj}])' * ones(N*D{jj}, 1)); %K x 1
    ff = ff + 0.5 * rauPhiErrSquare; %K vector
end
mogParam.ee = ee;
mogParam.ff = ff;
mogParam.xi     = ee ./ ff;
exLogRes.LogXi = psi(ee) - reallog(ff);
end

function [mogParam, lowRankRes, exErrSquare, exLogRes, sigma] = hdp_low_rank(noiseData, mogParam, lowRankRes, opts, exLogRes, sigma)
% Update U and V.

V = length(noiseData);
D = opts.DCell;   % V x 1 cell
N = opts.N;
r = opts.initDim;

%Update q(\lambda)
aa    = repmat(0.5 * cell2mat(D) + opts.a0, [1, r]);  % V x r
lproL = cellfun(@(x) x'*x, lowRankRes.L, 'UniformOutput', false);  % V x 1 cell
bb    = ones(V, r) * opts.b0;
for jj = 1:V
    bb(jj, :) = 0.5 * diag(lproL{jj}) + 0.5 * diag(sum(sigma.L{jj}, 3));
end
mogParam.aa = aa;
mogParam.bb = bb;
mogParam.lambda = aa ./ bb;
exLogRes.LogLambda = psi(aa) - reallog(bb);

exRjj = calculate_RiR(lowRankRes.R, sigma.R); % r x r x N tensor
resizeExRjj = reshape(exRjj, r^2, []);  %r^2 x N
phiXk = cellfun(@(x) x*mogParam.xi, mogParam.pphi, ...
                                                'UniformOutput', false);  % V x 1 cell, T x 1

%Update L: matlab
%TIter = opts.TIter;
%for vv = 1:V
    %for ii = 1:D{vv}
        %rauPhiXk = reshape(mogParam.rau{vv}(ii, :, :), [N, TIter{vv}]) * phiXk{vv}; % N x 1 vector
        %precisionVI = reshape(resizeExRjj * rauPhiXk, [r, r]) + diag(mogParam.lambda(vv, :));
        %sigma.L{vv}(:,:,ii) = precisionVI^(-1);

        %lowRankRes.L{vv}(ii, :) = (rauPhiXk' .* noiseData{vv}(ii, :)) *...
                                                       %lowRankRes.R' * sigma.L{vv}(:, :, ii);
    %end
%end

% Update L: C language
[lowRankRes.L, sigma.L] = updateL_C(mogParam.rau, mogParam.lambda, phiXk,...
        resizeExRjj, lowRankRes.R, noiseData);

exLii = cellfun(@(x, y) calculate_RiR(x',y), lowRankRes.L, sigma.L, ...
                                            'UniformOutput', false);  % V x 1 cell, r x r x D
% updata q(\tau)
cc = 0.5 * N + opts.c0;
dd = 0.5 * diag(lowRankRes.R * lowRankRes.R') + 0.5 * diag(sum(sigma.R, 3)) + opts.d0;
mogParam.cc = cc;
mogParam.dd = dd;
mogParam.tau = cc ./ dd;
exLogRes.LogTau = psi(cc) - reallog(dd);

% update R
resizeExLii = cellfun(@(x) reshape(x, r^2, []), exLii, ...
                                             'UniformOutput', false);  % V x 1 cell, r^2 x D
for jj = 1:N
    temp1 = zeros(1, r);
    temp = zeros(r, r);
    for vv = 1:V
        rauPhiXk2 = squeeze(mogParam.rau{vv}(:, jj, :)) * phiXk{vv};  % D x 1 matrix
        rauPhiXkLL = reshape(resizeExLii{vv} * rauPhiXk2, [r, r]);
        temp = temp + rauPhiXkLL;

        rauPhiXkData = rauPhiXk2'.*noiseData{vv}(:, jj)'; % 1 x D
        temp1 = temp1 + rauPhiXkData * lowRankRes.L{vv};
    end
    sigma.R(:, :, jj) = inv(temp + diag(mogParam.tau));
    lowRankRes.R(:, jj) = sigma.R(:, :, jj) * temp1';
end
exRjj = calculate_RiR(lowRankRes.R, sigma.R);  % r x r x N tensor

% calculate exErrSquare
matRjj = reshape(exRjj, [r^2, N]);  % r^2 x N matrix
clear exRjj;
XReconstruct = cellfun(@(x) x*lowRankRes.R, lowRankRes.L, ...
                                             'UniformOutput', false); % V x 1 cell, D x N
LRPro = cellfun(@(x) mtimesx(x',matRjj), resizeExLii, ...
                                    'UniformOutput', false);   % V x 1 cell, D x N matrix
clear matRjj;
exErrSquare = cellfun(@(x,y,z) x.^2+y-2*x.*z, noiseData, LRPro, XReconstruct, ...
                                          'UniformOutput', false);   % V x 1 cell, D x N
end

function exUiU = calculate_RiR(U, sigmaU)
%Input:
%   U: r x N matrix
%   sigmaU: r x r x N
%Output:
%   exUiU: r x r x N tensor

[r, N] = size(U);
UasTensor1 = reshape(U, [r,1,N]); %r x 1 x N
UasTensor2 = reshape(U, [1,r,N]);
exUiU = mtimesx(UasTensor1, UasTensor2) + sigmaU;
end

function [mogParam, exLogGauss, exLogRes, opts] = hdp_exp(mogParam, exErrSquare, exLogRes, opts)

K = opts.initK;
TCell = opts.TIter;   % V x 1 cell
D = opts.DCell;  % V x 1 cell
N = opts.N;

%calculate E[ln N(x_ij^v-L_i^v*R_j|0, \xi_k)]
xikErr = cellfun(@(x) bsxfun(@times, x, reshape(mogParam.xi, [1,1,K])), ...
                        exErrSquare, 'UniformOutput', false); % V x 1 cell, D x N x K
exLogGauss0 = cellfun(@(x) bsxfun(@minus, x, reshape(exLogRes.LogXi, [1,1,K])), ...
                        xikErr, 'UniformOutput', false); % V x 1 cell, D x N x K
exLogGauss = cellfun(@(x) -0.5*x-0.5*log(2*pi), exLogGauss0, 'UniformOutput', false);
clear xikErr exLogGauss0;

%Update q(C)
rauResize = cellfun(@(x, d, t) reshape(x, d*N, t), mogParam.rau, D, TCell,...
                                               'UniformOutput', false); % V x 1 cell, D*N x T
gaussResize = cellfun(@(x, d) reshape(x, d*N, K), exLogGauss, D, ...
                                              'UniformOutput', false);  % V x 1 cell, D*N x K
rauGauss = cellfun(@(x,y) x'*y, rauResize, gaussResize,...
                                                     'UniformOutput', false); % V cell, T x K
pphi0 = cellfun(@(x) bsxfun(@plus, x, exLogRes.LogBeta'), rauGauss, 'UniformOutput', false);
mogParam.pphi = cellfun(@(x) exp(bsxfun(@minus, x, logsumexp(x, 2))),...
                                                pphi0, 'UniformOutput', false);
clear rauResize pphi0;

%Update q(Z)
phiGauss = cellfun(@(x,y,t,d) reshape(x*y', d, [], t), gaussResize, mogParam.pphi, ...
                                   TCell, D, 'UniformOutput', false); % V x 1 cell, D x N x T
rau0 = cellfun(@(x,y,t) x+reshape(y, [1,1,t]), phiGauss, exLogRes.LogPi, ...
                                      TCell, 'UniformOutput', false); % V x 1 cell, D x N x T
mogParam.rau = cellfun(@(x) exp(bsxfun(@minus, x, logsumexp(x,3))), ...
                                                rau0, 'UniformOutput', false);

% cut Gaussian number for all the view
if opts.cutK && opts.iter > 1
    KScore = cellfun(@(x) sum(x, 1), mogParam.pphi,...
                                                 'UniformOutput', false); % V x 1 cell, 1 x K
    K_valid_all = cellfun(@(x) find(x>0)', KScore, 'UniformOutput', false); % V x 1 cell
    opts.ind_K_valid = unique(cell2mat(K_valid_all));
    opts.initK = length(opts.ind_K_valid);
    exLogGauss = cellfun(@(x) x(:, :, opts.ind_K_valid), exLogGauss, ...
                                                    'UniformOutput', false);
    mogParam.pphi = cellfun(@(x) x(:, opts.ind_K_valid), mogParam.pphi,...
                                                    'UniformOutput', false);
    mogParam.ssBeta1 = mogParam.ssBeta1(opts.ind_K_valid);
    mogParam.ssBeta2 = mogParam.ssBeta2(opts.ind_K_valid);
    exLogRes.LogBetaPie = exLogRes.LogBetaPie(opts.ind_K_valid);
    exLogRes.LogBetaPie1 = exLogRes.LogBetaPie1(opts.ind_K_valid);
    logBeta = exLogRes.LogBeta(opts.ind_K_valid);
    exLogRes.LogBeta = logBeta / sum(logBeta);
    mogParam.xi = mogParam.xi(opts.ind_K_valid);
    exLogRes.LogXi = exLogRes.LogXi(opts.ind_K_valid);
end
end

function bound = hdp_low_bound(mogParam, lowRankRes, opts, exLogGauss, exLogRes, sigma)
% Calculate the variation lower bound.
% exLogGauss: V x 1 cell, cell{v}: D x N x K

%V = length(mogParam.rau);
%N = opts.N;
%T = opts.initT;
%K = opts.initK;
%D = size(exLogGauss{1}, 1);
%r = size(sigma.R, 1);
%paramNew = mogParam;
%paramNew.rau = zeros(D, N, T, V);
%paramNew.pphi = zeros(V, T, K);
%paramNew.rrPi1 = zeros(V, T);
%paramNew.rrPi2 = zeros(V, T);
%exLogResNew = exLogRes;
%exLogResNew.LogPi = zeros(V, T);
%exLogResNew.LogPiPie = zeros(V, T);
%exLogResNew.LogPiPie1 = zeros(V, T);
%LNew = zeros(D, r, V);
%sigmaLNew = zeros(r, r, D, V);
%exLogGaussNew = zeros(D, N, V, K);
%for jj = 1:V
    %paramNew.rau(:, :, :, jj) = mogParam.rau{jj};
    %paramNew.pphi(jj, :, :) = reshape(mogParam.pphi{jj}, [1, T, K]);
    %exLogResNew.LogPi(jj, :) = exLogRes.LogPi{jj};
    %exLogResNew.LogPiPie(jj, :) = exLogRes.LogPiPie{jj};
    %exLogResNew.LogPiPie1(jj, :) = exLogRes.LogPiPie1{jj};
    %paramNew.rrPi1(jj, :) = mogParam.rrPi1{jj};
    %paramNew.rrPi2(jj, :) = mogParam.rrPi2{jj};
    %exLogGaussNew(:, :, jj, :) = reshape(exLogGauss{jj}, [D, N, 1, K]);
    %LNew(:, :, jj) = lowRankRes.L{jj};
    %sigmaLNew(:, :, :, jj) = sigma.L{jj};
%end
%lowRankRes.L = LNew;
%lowRankRes.sigmaL = sigmaLNew;
%lowRankRes.sigmaR = sigma.R;
%bound = low_bound_multi(paramNew, lowRankRes, opts, exLogGaussNew, exLogResNew);

K = opts.initK;
V = length(mogParam.rau);
TCell = opts.TIter;  % V x 1 cell
dim  = opts.initDim;
D = opts.DCell;  % V x 1 cell
DMat = cell2mat(D);  % V x 1 vector
N = opts.N;

bound = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%calculate E[q(W)log(p(X,W))]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rauResize = cellfun(@(x, t) reshape(x, [], t), mogParam.rau, TCell, ...
                                               'UniformOutput', false); % V x 1 cell, D*N x T
gaussResize = cellfun(@(x) reshape(x, [], K), exLogGauss, 'UniformOutput', false);
rauGauss = cellfun(@(x,y) x'*y, rauResize, gaussResize, 'UniformOutput', false);      % T x K
rauGaussPhi = cellfun(@(x,y) sum(reshape(x.*y, [], 1)), rauGauss, mogParam.pphi,...
                                                      'UniformOutput', true);  % V x 1 vector
bound = bound + sum(rauGaussPhi);
clear gaussResize;

%E[ln(p(L|\lambda))] and E[ln(p(R|\tau))]
exLrr = zeros(V, dim);
temp = cellfun(@(x) x'*x, lowRankRes.L, 'UniformOutput', false); % V x 1 cell
for jj = 1:V
    exLrr(jj, :) = diag(temp{jj})+diag(sum(sigma.L{jj}, 3));
end
exRrr = diag(lowRankRes.R * lowRankRes.R') + diag(sum(sigma.R, 3)); % dim x 1 vector
bound = bound + 0.5*sum(sum(repmat(DMat, [1, dim]).*exLogRes.LogLambda ...
                               - repmat(DMat*log(2*pi), [1, dim]) - mogParam.lambda.*exLrr));
bound = bound + 0.5*sum(N*exLogRes.LogTau - N*log(2*pi) - mogParam.tau.*exRrr);
clear temp exLrr exRrr;

%E[ln(p(\lambda))]
bound = bound + sum(sum(opts.a0*log(opts.b0) - gammaln(opts.a0)...
                            + (opts.a0-1)*exLogRes.LogLambda - opts.b0*mogParam.lambda));

%E[\ln(p(\tau))]
bound = bound + sum(opts.c0*log(opts.d0) - gammaln(opts.c0) + (opts.c0-1)*exLogRes.LogTau...
                                                                 - opts.d0*mogParam.tau);

%E[ln(p(C))]
phiExLogBeta = cellfun(@(x) sum(x*exLogRes.LogBeta), mogParam.pphi,...
                                                'UniformOutput', true); % V x 1 vector
bound = bound + sum(phiExLogBeta);

%E[ln(p(Beta^{'}))]
bound = bound + sum(exLogRes.LogGamma + (mogParam.ggamma-1)*exLogRes.LogBetaPie1);

%E[ln(p(gamma))]
bound = bound + opts.g0*log(opts.h0) - gammaln(opts.g0) + (opts.g0-1)*exLogRes.LogGamma ...
                                                              - opts.h0*mogParam.ggamma;

%E[ln(p(Z))]
rauExLogPi = cellfun(@(x,y) sum(x*y'), rauResize, exLogRes.LogPi, 'UniformOutput', true);
bound = bound + sum(rauExLogPi);
clear rauResize;

%E[ln(p(pi^{'}))]
%mogParam.alpha: V x 1 vector
for jj = 1:V
    bound = bound + sum(exLogRes.LogAlpha(jj)+(mogParam.alpha(jj)-1)*exLogRes.LogPiPie1{jj});
end

%E[ln(p(alpha))]
bound = bound + sum(opts.m0*log(opts.n0) - gammaln(opts.m0) +...
                                     (opts.m0-1)*exLogRes.LogAlpha - opts.n0*mogParam.alpha);

%E[p(\xi)]
bound = bound + sum(opts.e0*log(opts.f0) - gammaln(opts.e0) ...
                                        + (opts.e0-1)*exLogRes.LogXi - opts.f0*mogParam.xi);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%lcalculate E[q(W)log(q(W))]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%E[ln(q(L))]
for jj = 1:V
    for ii= 1:D{jj}
        cholLi = chol(sigma.L{jj}(:, :, ii));
        logDetLi = sum(reallog(diag(cholLi)));
        bound = bound + 0.5*dim*(1+log(2*pi)) + logDetLi;
    end
end

%E[ln(q(R))]
for jj=1:N
    cholRj = chol(sigma.R(:,:,jj));
    logDetRj = sum(reallog(diag(cholRj)));
    bound = bound + 0.5*dim*(1+log(2*pi)) + logDetRj;
end

%E[ln(q(\lambda))]
bound = bound + sum(sum(mogParam.aa - reallog(mogParam.bb) + gammaln(mogParam.aa) ...
                                                    + (1-mogParam.aa) .* psi(mogParam.aa)));

%E[log(q(\tau))]
bound = bound + sum(mogParam.cc - reallog(mogParam.dd) + gammaln(mogParam.cc) ...
                                                     + (1-mogParam.cc) .* psi(mogParam.cc));

%E[q(\xi)]
bound = bound + sum(mogParam.ee - reallog(mogParam.ff) + gammaln(mogParam.ee)...
                                                    + (1-mogParam.ee) .* psi(mogParam.ee));

%E[lnq(C)]
logPhi = cellfun(@get_log, mogParam.pphi, 'UniformOutput', false);
phiLogPhi = cellfun(@(x,y) sum(reshape(x.*y,[],1)), mogParam.pphi,...
                                                            logPhi, 'UniformOutput', true);
bound = bound - sum(phiLogPhi);
clear logPhi;

%E[lnq((\beta^{'}))]
bLogss = betaln(mogParam.ssBeta1, mogParam.ssBeta2);
entropyBetaPie = bLogss - (mogParam.ssBeta1-1) .* psi(mogParam.ssBeta1) -...
  (mogParam.ssBeta2-1) .* psi(mogParam.ssBeta2) + (mogParam.ssBeta1+mogParam.ssBeta2-2)...
                                                 .* psi(mogParam.ssBeta1+mogParam.ssBeta2);
bound = bound + sum(entropyBetaPie(:));

%E[lnq(gamma)]
bound = bound + mogParam.gg - reallog(mogParam.hh) + gammaln(mogParam.gg)...
                                                        + (1-mogParam.gg)*psi(mogParam.gg);

%E[lnq(Z)]
logRau = cellfun(@get_log, mogParam.rau, 'UniformOutput', false);
rauLogRau = cellfun(@(x,y) sum(reshape(x.*y,[],1)), mogParam.rau,...
                                                            logRau, 'UniformOutput', true);
bound = bound - sum(rauLogRau);
clear rauLogRau logRau;

%E[lnq(pi^{'})]
for jj = 1:V
    bLog = betaln(mogParam.rrPi1{jj}, mogParam.rrPi2{jj});
    entropyPi = bLog - (mogParam.rrPi1{jj}-1) .* psi(mogParam.rrPi1{jj})...
                                    - (mogParam.rrPi2{jj}-1) .* psi(mogParam.rrPi2{jj})...
                                            + (mogParam.rrPi1{jj}+mogParam.rrPi2{jj}-2)...
                              .* psi(mogParam.rrPi1{jj}+mogParam.rrPi2{jj}); %1 x T matrix
    bound = bound + sum(entropyPi(:));
end
clear entropyPi;

%E[ln(q(\alpha))]
bound = bound + sum(mogParam.mm - reallog(mogParam.nn) + gammaln(mogParam.mm)...
                                                     + (1-mogParam.mm).*psi(mogParam.mm));
end

function logpphi = get_log(pphi)
logpphi = log(pphi);
logpphi(isinf(logpphi)) = 0;
end

function res = merge_T(pphi, thres)
% Input:
%   pphi: T x K matrix
% Output:
%   res: cell, each cell element contains the index of Gaussian components need to be merged.

T = size(pphi, 1);
flag = true(T, 1);
iter = 0;
res = cell(T, 1);
while any(flag)
    iter = iter + 1;
    ind_true = find(flag);
    if sum(ind_true) > 1
        sub_pphi = pphi(ind_true, :);
        dist = max(abs(bsxfun(@minus, sub_pphi(1, :), sub_pphi)), [], 2);
        res{iter} = ind_true(dist<thres);
    else
        res{iter} = ind_true;
    end
    flag(res{iter}) = false;
end
res = res(1:iter);
assert(sum(cellfun(@length, res)) == T);
end

