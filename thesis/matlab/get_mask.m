function mask = get_mask(W, H, sampling_factor, start_line, keep_center)
%GET_MASK Summary of this function goes here
%   Detailed explanation goes here
    mask = zeros(W, H);
    mask(start_line:sampling_factor:end, :) = 1;

    center_line = floor(H/2);
    center_width = floor(0.5 * keep_center * H);
    mask(center_line-center_width+1 : 1 : center_line+center_width, :) = 1;

end

