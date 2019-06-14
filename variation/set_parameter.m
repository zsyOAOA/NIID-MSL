function opt = set_parameter(varargin)
%set_parameters.m set parameters of MoG Model,return a structure containing complete parameter setting
% usage:
%      opt = set_parameters('Param1','StringValue1','Param2',NumValue2)
% Output:
%      opt: struct of hyperparameter.

opt = struct('a0', 1e-4,...
             'b0', 1e-4,...      %\lambda_L^v \sim Gamma(a0, b0)
             'c0', 1e-4,...
             'd0', 1e-4,...      %\lambda_R \sim Gamma(c0, d0)
             'e0', 1e-4,...
             'f0', 1e-4,...      %\xi_k \sim Gamma(e0, f0)
             'g0', 1e-4,...
             'h0', 1e-4,... %\gamma \sim Gamma(g0, h0), scale parameter of Global Dirichlet Process
             'm0', 1e-4,...
             'n0', 1e-4,...  %\alpha_v \sim Gamma(m0, n0), scale parameter of local Dirichlet Process
             'initT', 15,...     %Initialization number of local MoG components
             'initK', 25,...
             'initDim', 5,...
             'tol', 1e-5,...
             'itermax', 20,...
             'display', 1,...
             'bound', 0,...
             'cutK', 1,...
             'mergeT', 1,...
             'merge_thres', 0.01, ...
             'init',2);          %1:randn, 2: PCA, 3: RPCA, 4: SVD

numparameters = nargin;
if numparameters == 0
    return
end
if rem(numparameters, 2) ~= 0
    error('The paarmeters must be field-value pairs')
end
if rem(numparameters, 2) == 0 && numparameters > 1
    for i = 1:2:numparameters
        opt.(varargin{i}) = varargin{i+1};
    end
end

if opt.initK < opt.initT
    error('initT < initK must be satisfied!');
end
if opt.bound
    opt.mergeT = 0;
end

end

