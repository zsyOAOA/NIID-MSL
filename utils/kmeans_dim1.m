function [center, ind_data] = kmeans_dim1(data, K, iter_max)
% Kmeans for 1 dimension.
% Input:
%   data: n x 1 vector
%   K: number of clusters
%   iter_max: maximal iteration
% Output:
%   center: K x 1 vector
%   ind_data: n x 1 vector, cluster indicator

n = length(data);
shuffle = randperm(n);
center = data(shuffle(1:K));

for ii = 1:iter_max
    dist = bsxfun(@minus, data, center').^2;  % n x K
    [~, ind_data] = min(dist, [], 2);
    mask = full(sparse(1:n, ind_data, 1, n, K, n));
    center_t = sum(bsxfun(@times, data, mask), 1) ./ sum(mask, 1);  % 1 x K
    center = center_t';  % K x 1
end

end

