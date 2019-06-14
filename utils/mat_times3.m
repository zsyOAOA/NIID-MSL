function C = mat_times3(A, B)
% Input:
%   A: m x n x v tensor
%   B: n x d matrix
% Output:
%   C: m x d x V tensor, C(:, :, vv) = A(:, :, vv) * B

[m, n, v] = size(A);
A_mat = reshape(permute(A, [1,3,2]), [m*v, n]);  % m*v x n matrix
C_mat = A_mat * B;
C = permute(reshape(C_mat, m, v, []), [1, 3, 2]);
end
