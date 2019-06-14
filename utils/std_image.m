function Y = std_image(im)
% This Function perform data standardization,x-X-min/(x_max-x_min),every column represent one data sample
%Input:
%   im: m x n image data matrix
%Output:
%   Y: m x n stardardzied data matrix
x_min = min(min(im));
x_max = max(max(im));
Y = (im - x_min) ./ (x_max - x_min);
Y(isnan(Y)) = 0;

