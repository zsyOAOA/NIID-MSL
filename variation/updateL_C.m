function [L_new, sigmaLNew] = updateL_C(rau, lambda, phiXk, resizeExRjj, R, noiseData)
% Update L using matlab and C language.
% Input:
%   rau, V x 1 cell
%   labmda, V x r matrix
%   phiXk, V x 1 cell, T x 1
%   resizeExRjj, r^2 x N
%   R, r x N matrix
%   noiseData, V x 1 cell, D x N
% Output:  
%   L_new, V x 1 cell, D x r matrix
%   simgaLNew, V x 1 cell, r x r x D matrix
