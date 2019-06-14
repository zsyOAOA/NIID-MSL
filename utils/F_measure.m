function [val_best, thres_best, val, thres] = F_measure(img, mask_ref)
% Calculate the F measure.
% Input:
%   img: h x w x c tensor
%   mask_ref: h x w matrix, groundtruth
% Output:
%   val_best: F-measure value
%   thres_best: threshold
%   val, thres: F-measure for different threshold

thres = 0:0.0001:0.3;
val = zeros(1, length(thres));

for ii = 1:length(thres)
    mask0 = sum(double(img > thres(ii)), 3);
    mask = double(mask0 >= 1);
    if sum(mask(:)) == 0 || sum(mask_ref(:)) == 0
        mask = double(~mask);
        mask_ref = double(~mask_ref);
    end
    up = sum((mask(:)+mask_ref(:)) == 2);
    precision = up / sum(mask(:));
    recall = up / sum(mask_ref(:));
    val(ii) = 2 * precision*recall / (precision+recall);
end

[val_best, ind_best]= max(val);
thres_best = thres(ind_best);

