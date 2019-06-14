function R = find_close_one(X, k, m)
%Input:
  %X: d x n matrix
  %m: d x k matrix
%Output:
  %R: n x K matrix, n is the number of data samples.

n = size(X, 2);
[~, ind] = max(bsxfun(@minus, m'*X, dot(m,m,1)'/2), [], 1);
[u, ~, label] = unique(ind);
while k ~= length(u)
    idx = randsample(n,k);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    [u,~,label] = unique(label);
end
R = full(sparse(1:n, label, 1, n, k, n));
end
