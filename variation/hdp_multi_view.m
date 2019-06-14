function [mogParam, lowRankRes, recover] = hdp_multi_view(noiseData, opts)
%Input:
%   noiseData: D X N X V tensor data, N is number of data samples, D is the dimensionality
%   opts: struct, returned by function set_parameter.
%Output:
%   mogParam: MoG parameter, struct.
%       mogParam.rau: D x N x T x V tensor, responsibility matrix of local MoG.
%       mogParam.pphi: V x T x K tensor, responsibility matrix of Global MoG.
%   lowRankRes: Low Rank parameter. struct.
%       lowRankRes.L: D x r x V tensor
%       lowRankRes.R: r x N matrix
%       lowRankRes.sigmaL: r x r x D x V tensor
%       lowRankRes.sigmaR: r x r x N tensor
%   recover: D x N x V tensor.

%Initialization
[mogParam, lowRankRes, exErrSquare] = hdp_multi_init(noiseData, opts);

iter = 1;
currentTol = Inf;
bound = -Inf(opts.itermax, 1);
exLogRes = struct();
while (iter <= opts.itermax)
    [mogParam, exLogRes] = hdp_multi_maxv(mogParam, opts, exErrSquare, exLogRes);
    [mogParam, lowRankRes, exErrSquare, exLogRes] = hdp_multi_low_rank(noiseData, mogParam,...
        lowRankRes, opts, exLogRes);
    [mogParam, exLogGauss] = hdp_multi_exp(mogParam, exErrSquare, exLogRes, opts);
    if opts.display
        if opts.bound
            bound(iter) = low_bound_multi(mogParam, lowRankRes, opts, exLogGauss, exLogRes);
            if iter > 1
                currentTol = abs(bound(iter)-bound(iter-1)) / abs(bound(iter-1));
            end
            fprintf('Iter=%2d, bound=%.4f, tol=%.6f...\n', iter, bound(iter), currentTol);
        else
            fprintf('Iter=%2d...\n', iter);
        end
    end
    if opts.bound && (currentTol < opts.tol)
        break;
    end
    iter = iter + 1;
end
mogParam.bound = bound(1:iter-1);
recover = mtimesx(lowRankRes.L, lowRankRes.R);
end

function [mogParam, lowRankRes, exErrSquare] = hdp_multi_init(noiseData, opts)
% exErrSquare: D x N x V matrix

K         = opts.initK;
T         = opts.initT;
r       = opts.initDim;
[D, N, V] = size(noiseData);
scale2    = sum(noiseData(:).^2)/(D*N*V);
scale     = sqrt(scale2);

switch opts.init
    case 1
        L  = randn(D, r, V) * sqrt(scale);
        R  = randn(r, N) * sqrt(scale);
        XReconstruct = mtimesx(L, R); % D x N x V tensor
        err= noiseData - XReconstruct;  % D x N x V tensor
        if opts.display
            disp('Randomly Initialization.');
        end
    case 2
        noiseResize = reshape(permute(noiseData, [1, 3, 2]), [D*V, N]);
        [LAll, R0, ~,~] = PCA(noiseResize, r);
        R = R0';  % R: r x N
        L = permute(reshape(LAll, [D, V, r]), [1, 3, 2]); % D x r x V
        XReconstruct = mtimesx(L, R); % D x N x V tensor
        err= noiseData - XReconstruct;  % D x N x V tensor
        if opts.display
            disp('PCA Initialization.');
        end
    case 3
        noiseResize = reshape(permute(noiseData, [1, 3, 2]), [D*V, N]);
        [A_hat, ~, ~] = inexact_alm_rpca(noiseResize, -1, -1, 100);
        [LAll, R0, ~,~] = PCA(A_hat, r);
        L = permute(reshape(LAll, [D, V, r]), [1, 3, 2]); % D x r x V
        R = R0';  % R: r x N
        XReconstruct = mtimesx(L, R); % D x N x V tensor
        err= noiseData - XReconstruct;  % D x N x V tensor
        if opts.display
            disp('RPCA Initialization.');
        end
end
sigmaL            = repmat(eye(r)*scale, [1, 1, D, V]);
sigmaR            = repmat(eye(r)*scale, [1, 1, N]);
lowRankRes.L      = L;
lowRankRes.sigmaL = sigmaL;
lowRankRes.R      = R;
lowRankRes.sigmaR = sigmaR;
exErrSquare       = err.^2;

errMat = reshape(err, [], V);
rau0 = zeros(D*N, T, V);
pphi0 = zeros(T, K, V);

samplek = datasample(errMat(:), K, 'Replace', false); %K x 1 vector

for bb = 1:V
    indexT = randsample(K, T);
    pphi0(:,:,bb) = full(sparse(1:T, indexT, 1, T, K, T));

    m = samplek(indexT);  %T x 1
    [~, labelT] = max(bsxfun(@minus, m*errMat(:, bb)', 0.5*m.^2), [], 1);

    rau0(:, :, bb) = full(sparse(1:D*N, labelT, 1, D*N, T, D*N));
end

%for bb = 1:V
    %indexT = randsample(K, T);
    %pphi0(:,:,bb) = full(sparse(1:T, indexT, 1, T, K, T));
    %rau0(:,:,bb) = find_close_one(errMat(:, bb)', T, samplek(indexT)');
%end

mogParam.rau =reshape(rau0, [D, N, T, V]); % D x N x T x V
mogParam.pphi = permute(pphi0, [3, 1, 2]); % V x T x K
end

function [mogParam, exLogRes] = hdp_multi_maxv(mogParam, opts, exErrSquare, exLogRes)
%   mogParam: MoG parameters
%       mogParam.alpha: global scale parameter of DP
%       mogParam.rrPi1; mogParam.rrPi2;
%       mogParam.ggamma: local scale parameter of DP
%       mogParam.xi: precision of noise
%   exLogRes:
%       exLogRes.LogPiPie: E[log({\Pi_t^v}^{'})]
%       exLogRes.LogPiPie1: E[1-log({\Pi_t^v}^{'})]
%       exLogRes.LogBetaPie: E[log({\beta_k}^{'})]
%       exLogRes.LogBetaPie1: E[1-log({\beta_k}^{'})]
%       exLogRes.LogXi: E[log(\xi_k)]
%       exLogRes.LogLambda: E[log(\lambda)]
%       exLogRes.LogGamma: E{log(\gamma)}
%       exLogRes.LogAlpha: E[log(\alpha)]
%

[D, N, V] = size(exErrSquare);
T = opts.initT;
K = opts.initK;

%Update q(\pi')
numVT = sum(sum(permute(mogParam.rau, [4, 3, 1, 2]), 4), 3); % V x T matrix
rrPi1 = numVT + 1;  % V x T matrix
if ~isfield(mogParam, 'alpha')
    mogParam.alpha = opts.m0 /opts.n0 * ones(V, 1);
end
rrPi2              = bsxfun(@minus, sum(numVT,2), cumsum(numVT,2)) + repmat(mogParam.alpha, [1, T]); % V x T matrix
temp               = psi(rrPi1 + rrPi2);
mogParam.rrPi1 = rrPi1;
mogParam.rrPi2 = rrPi2;
exLogRes.LogPiPie  = psi(rrPi1) - temp; % V x T matrix
exLogRes.LogPiPie1 = psi(rrPi2) - temp; % V x T matrix
cumExLogPiPie1 = [zeros(V, 1) cumsum(exLogRes.LogPiPie1(:, 1:(end-1)), 2)];
exLogRes.LogPi = exLogRes.LogPiPie + cumExLogPiPie1;
exLogRes.LogPi = bsxfun(@rdivide, exLogRes.LogPi, sum(exLogRes.LogPi, 2));
clear cumExLogPiPie1;

%Update q(\alpha)
mm             = T + opts.m0;
nn             = opts.n0 - sum(exLogRes.LogPiPie1, 2); % V vector
mogParam.alpha = mm ./ nn;
mogParam.mm = mm;
mogParam.nn = nn;
exLogRes.LogAlpha = psi(mm) - reallog(nn);

%Update q(beta')
ssBeta1 = reshape(mogParam.pphi, [], K)' * ones(V*T, 1) + 1; % K x 1 vector
if ~isfield(mogParam, 'ggamma')
    mogParam.ggamma = opts.g0 / opts.h0;
end
ssBeta2              = sum(ssBeta1) - cumsum(ssBeta1) + mogParam.ggamma;
temp                 = psi(ssBeta1 + ssBeta2);
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
phiResize       = reshape(permute(mogParam.pphi, [3, 2, 1]), K, []); %K x (TV) matrix
numDataK        = phiResize * reshape(mogParam.rau, [D*N, T*V])' * ones(D*N, 1);   %K x 1 vector
ee              = 0.5*numDataK +opts.e0; %K x 1 vector
rauErrSquare    = bsxfun(@times, mogParam.rau, reshape(exErrSquare, [D, N, 1, V])); % D x N x T x V tensor
rauPhiErrSquare = phiResize * reshape(rauErrSquare, [N*D, T*V])' * ones(N*D, 1); %K x 1 vector
ff              = opts.f0 + 0.5 * rauPhiErrSquare; %K vector
mogParam.ee = ee;
mogParam.ff = ff;
mogParam.xi     = ee ./ ff;
exLogRes.LogXi = psi(ee) - reallog(ff);
end

function [mogParam, lowRankRes, exErrSquare, exLogRes] = hdp_multi_low_rank(noiseData, mogParam, lowRankRes, opts, exLogRes)
% Update U and V.
% Output:
%   exErrSquare: D x N x V tensor, E[(x_{ij}^2-{L_{i,:}^v}^T*R_{j,:})^2]
%   exLogRes:
%       exLogRes.LogLambda: E[log(\lambda)]
%       exLogRes.LogTau: E[log(\tau)]

[D, N, V] = size(noiseData);
r       = size(lowRankRes.L, 2);
[~, T, K] = size(mogParam.pphi);

%Update q(\lambda)
aa    = 0.5 * D + opts.a0;
lproL = mtimesx(permute(lowRankRes.L, [2, 1, 3]), lowRankRes.L); % r x r x V tensor
bb    = zeros(V, r);
for jj = 1:V
    bb(jj, :) = 0.5 * diag(lproL(:, :, jj)) + 0.5 * diag(sum(lowRankRes.sigmaL(:, :, :, jj), 3)) + opts.b0;
end
mogParam.aa = aa;
mogParam.bb = bb;
mogParam.lambda = aa ./ bb;
exLogRes.LogLambda = psi(aa) - reallog(bb);

phiXk0 = reshape(mogParam.pphi, [], K) * mogParam.xi;
phiXk = reshape(phiXk0, V, T); % V x T matrix
clear phiXk0;

%Update L
exRjj = calculate_RiR(lowRankRes.R, lowRankRes.sigmaR); % r x r x N tensor
resizeExRjj = reshape(exRjj, r^2, []);  %r^2 x N
for jj = 1:V
    for ii = 1:D
        rauPhiXk = permute(mogParam.rau(ii, :, :, jj), [2, 3, 1, 4]) * phiXk(jj, :)'; % N x 1 vector
        precisionVI = reshape(resizeExRjj * rauPhiXk, [r, r]) + diag(mogParam.lambda(jj, :));
        lowRankRes.sigmaL(:,:,ii,jj) = precisionVI^(-1);

        lowRankRes.L(ii, :, jj) = (rauPhiXk' .* noiseData(ii, :, jj)) * lowRankRes.R' * lowRankRes.sigmaL(:, :, ii, jj);
    end
end
clear rauPhiXk exRjj resizeExRjj;
exLii = calculate_RiR(permute(lowRankRes.L, [2, 1, 3]), lowRankRes.sigmaL); % r x r x D x V tensor

% updata q(\tau)
cc           = 0.5 * N + opts.c0;
dd           = 0.5 * diag(lowRankRes.R * lowRankRes.R') + 0.5 * diag(sum(lowRankRes.sigmaR, 3)) + opts.d0;
mogParam.cc = cc;
mogParam.dd = dd;
mogParam.tau = cc ./ dd;
exLogRes.LogTau = psi(cc) - reallog(dd);

resizeExLii = reshape(exLii, r^2, []);  % (r^2) x (D*V)
% update R
for jj = 1:N
    rauPhiXk2 = sum(bsxfun(@times, permute(mogParam.rau(:, jj, :, :), [1, 4, 3, 2]),...
                        reshape(phiXk, [1, V, T])), 3);   % D x V matrix
    precisionjj = reshape(resizeExLii * rauPhiXk2(:), [r, r]) + diag(mogParam.tau);
    lowRankRes.sigmaR(:, :, jj) = precisionjj^(-1);

    rauPhiXkData = rauPhiXk2 .* reshape(noiseData(:, jj, :), [D, V]);  % D x V matrix
    lowRankRes.R(:, jj) = lowRankRes.sigmaR(:, :, jj) * ...
                               (reshape(permute(lowRankRes.L, [2,1,3]), r, []) * rauPhiXkData(:));
end
clear resizeExLii rauPhiXk2 rauPhiXkData;
exRjj = calculate_RiR(lowRankRes.R, lowRankRes.sigmaR);  % r x r x N tensor

% calculate exErrSquare
matLii0= reshape(exLii, [r^2, D, V]); %r^2 x D x V tensor
clear exLii;
matLii = permute(matLii0, [2, 1, 3]);  % D x r^2 x V tensor
clear matLii0;
matRjj = reshape(exRjj, [r^2, N]);  % r^2 x N matrix
clear exRjj;
XReconstruct = mtimesx(lowRankRes.L, repmat(lowRankRes.R, [1, 1, V]));
LRPro = mtimesx(matLii, repmat(matRjj, [1, 1, V]));  % D x N x V tensor
clear matLii matRjj;
exErrSquare = noiseData.^2 + LRPro - 2 * noiseData.*XReconstruct;
clear LRPro XReconstruct;

end

function exUiU = calculate_RiR(U, sigmaU)
%Input:
%   U: r x N matrix or r x N x V tensor
%Output:
%   exUiU: r x r x N or r x r x N x V tensor

if length(size(U)) == 2
    [r, N] = size(U);
    UasTensor1 = reshape(U, [r,1,N]); %r x 1 x N
    UasTensor2 = reshape(U, [1,r,N]);
    exUiU = mtimesx(UasTensor1, UasTensor2) + sigmaU;
else
    [r, N, V] = size(U);
    UasTensor1 = reshape(U, [r, 1, N, V]); %r x 1 x N x V
    UasTensor2 = reshape(U, [1, r, N, V]);
    exUiU = mtimesx(UasTensor1, UasTensor2) + sigmaU;
end
end

function [mogParam, exLogGauss] = hdp_multi_exp(mogParam, exErrSquare, exLogRes, opts)

[D, N, ~, ~] = size(mogParam.rau);
[V, T, K] = size(mogParam.pphi);

%calculate E[ln N(x_ij^v-L_i^v*R_j|0, \xi_k)]
xikErr = bsxfun(@times, exErrSquare, reshape(mogParam.xi, [1, 1, 1, K])); %D x N x V x K tensor
exLogGauss = -0.5*(bsxfun(@minus, xikErr, reshape(exLogRes.LogXi, [1, 1, 1, K]))) - 0.5 * log(2*pi); % D x N x V x K tensor
clear xikErr;

%Update q(C)
rauResize = reshape(mogParam.rau, [D*N, T, V]); % D*N x T x V tensor
gaussResize = reshape(exLogGauss, [D*N, V, K]); % D*N x V x K tensor
pphi0 = zeros(T, K, V);
for jj = 1:V
    pphi0(:,:,jj) = rauResize(:, :, jj)' * permute(gaussResize(:, jj, :), [1, 3, 2]);  % T x K matrix
end
clear rauResize;
pphi0 = permute(pphi0, [3,1,2]);%V x T x K matrix
pphi0 = bsxfun(@plus, pphi0, reshape(exLogRes.LogBeta, [1,1,K]));
pphi0 = exp(bsxfun(@minus, pphi0, logsumexp(pphi0, 3)));
mogParam.pphi = pphi0;

%Update q(Z)
rau0 = zeros(D*N, T, V); % D*N x T x V
for jj = 1:V
    rau0(:, :, jj) = permute(gaussResize(:, jj, :), [1,3,2]) * permute(mogParam.pphi(jj,:,:), [3,2,1]);
end
rau0 = permute(rau0, [1,3,2]); % D*N x V x T
rau0 = bsxfun(@plus, rau0, reshape(exLogRes.LogPi, [1, V, T])); %D*N x V x T
rau0 = exp(bsxfun(@minus, rau0, logsumexp(rau0, 3))); % D*N x V x T
rau0 = permute(rau0, [1,3,2]); % D*N x T x V
mogParam.rau = reshape(rau0, [D, N, T, V]);
clear rau0 pphi0;

end

function bound = low_bound_multi(mogParam, lowRankRes, opts, exLogGauss, exLogRes)
% Calculate the variation lower bound.

[V, T, K] = size(mogParam.pphi);
[D, dim, ~]  = size(lowRankRes.L);
N         = size(lowRankRes.R, 2);

bound = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%calculate E[q(W)log(p(X,W))]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rauResize = reshape(mogParam.rau, [D*N, T, V]);
gaussResize = reshape(exLogGauss, [D*N, V, K]);
phiGauss = zeros(D*N, T, V);
for jj = 1:V
    phiGauss(:,:,jj) = permute(gaussResize(:, jj, :), [1,3,2]) * permute(mogParam.pphi(jj,:,:), [3,2,1]);
end
rauPhiGauss = phiGauss .* rauResize; %D*N x T x V
bound = bound + sum(rauPhiGauss(:));
clear phiGauss rauPhiGauss gaussResize;

%E[ln(p(L|\lambda))] and E[ln(p(R|\tau))]
exLrr = zeros(V, dim);
temp = mtimesx(permute(lowRankRes.L, [2, 1, 3]), lowRankRes.L);  % dim x dim x V
for jj = 1:V
    exLrr(jj, :) = diag(temp(:, :, jj))+diag(sum(lowRankRes.sigmaL(:, :, :, jj), 3));
end
exRrr = diag(lowRankRes.R * lowRankRes.R') + diag(sum(lowRankRes.sigmaR, 3)); % dim x 1 vector
bound = bound + 0.5*sum(sum(D*exLogRes.LogLambda - D*log(2*pi) - mogParam.lambda.*exLrr));
bound = bound + 0.5*sum(N*exLogRes.LogTau - N*log(2*pi) - mogParam.tau.*exRrr);
clear temp exLrr exRrr;

%E[ln(p(\lambda))]
bound = bound + sum(sum(opts.a0*log(opts.b0) - gammaln(opts.a0) + (opts.a0-1)*exLogRes.LogLambda...
        - opts.b0*mogParam.lambda));

%E[\ln(p(\tau))]
bound = bound + sum(opts.c0*log(opts.d0) - gammaln(opts.c0) + (opts.c0-1)*exLogRes.LogTau...
              - opts.d0*mogParam.tau);

%E[ln(p(C))]
phiExLogBeta = reshape(mogParam.pphi, [], K) * exLogRes.LogBeta;
bound = bound + sum(phiExLogBeta);
clear phiExLogBeta;

%E[ln(p(Beta^{'}))]
bound = bound + sum(exLogRes.LogGamma + (mogParam.ggamma-1)*exLogRes.LogBetaPie1);

%E[ln(p(gamma))]
bound = bound + opts.g0*log(opts.h0) - gammaln(opts.g0) + (opts.g0-1)*exLogRes.LogGamma ...
                - opts.h0*mogParam.ggamma;

%E[ln(p(Z))]
%if iter < 3
    rauExLogPi = bsxfun(@times, permute(rauResize, [1, 3, 2]), reshape(exLogRes.LogPi, [1,V,T])); % D*N x V x T
    bound = bound + sum(rauExLogPi(:));
    clear rauExLogPi rauResize;

%E[ln(p(pi^{'}))]
    alphaExPi = bsxfun(@times, mogParam.alpha-1, exLogRes.LogPiPie1);
    bound = bound + sum(sum(bsxfun(@plus, alphaExPi, exLogRes.LogAlpha)));
    clear alphaExPi;

%E[ln(p(alpha))]
    bound = bound + sum(opts.m0*log(opts.n0) - gammaln(opts.m0) + (opts.m0-1)*exLogRes.LogAlpha...
                - opts.n0*mogParam.alpha);
%end

%E[p(\xi)]
bound = bound + sum(opts.e0*log(opts.f0) - gammaln(opts.e0) ...
        + (opts.e0-1)*exLogRes.LogXi - opts.f0*mogParam.xi);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%lcalculate E[q(W)log(q(W))]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%E[ln(q(L))]
for jj = 1:V
    for ii= 1:D
        cholLi = chol(lowRankRes.sigmaL(:, :, ii, jj));
        logDetLi = sum(reallog(diag(cholLi)));
        bound = bound + 0.5*dim*(1+log(2*pi)) + logDetLi;
    end
end

%E[ln(q(R))]
for jj=1:N
    cholRj = chol(lowRankRes.sigmaR(:,:,jj));
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
bound = bound + sum(mogParam.ee - reallog(mogParam.ff) + gammaln(mogParam.ee) + (1-mogParam.ee) .* psi(mogParam.ee));

%E[lnq(C)]
logPhi = log(mogParam.pphi);
logPhi(isinf(logPhi)) = 0;
phiLogPhi = mogParam.pphi .* logPhi;
bound = bound - sum(phiLogPhi(:));
clear phiLogPhi logPhi;

%E[lnq((\beta^{'}))]
bLogss = betaln(mogParam.ssBeta1, mogParam.ssBeta2);
entropyBetaPie = bLogss - (mogParam.ssBeta1-1) .* psi(mogParam.ssBeta1) - (mogParam.ssBeta2-1) .* psi(mogParam.ssBeta2) ...
                 + (mogParam.ssBeta1+mogParam.ssBeta2-2) .* psi(mogParam.ssBeta1+mogParam.ssBeta2);
bound = bound + sum(entropyBetaPie(:));
clear bLogss entropyBetaPie;

%E[lnq(gamma)]
bound = bound + mogParam.gg - reallog(mogParam.hh) + gammaln(mogParam.gg)...
              + (1-mogParam.gg)*psi(mogParam.gg);

%if iter < 3
    %E[lnq(Z)]
    logRau = log(mogParam.rau);
    logRau(isinf(logRau)) = 0;
    rauLogRau = mogParam.rau .* logRau;
    bound = bound - sum(rauLogRau(:));
    clear rauLogRau logRau;

    %E[lnq(pi^{'})]
    bLog = betaln(mogParam.rrPi1, mogParam.rrPi2);
    entropyPi = bLog - (mogParam.rrPi1-1) .* psi(mogParam.rrPi1) - (mogParam.rrPi2-1) .* psi(mogParam.rrPi2) ...
                + (mogParam.rrPi1+mogParam.rrPi2-2) .* psi(mogParam.rrPi1+mogParam.rrPi2); %V x T matrix
    bound = bound + sum(entropyPi(:));
    clear entropyPi;

%E[ln(q(\alpha))]
    bound = bound + sum(mogParam.mm - reallog(mogParam.nn) + gammaln(mogParam.mm)...
            + (1-mogParam.mm).*psi(mogParam.mm));
end

