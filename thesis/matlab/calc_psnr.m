function psnr = calc_psnr(mse)
%PSNR Summary of this function goes here
%   Detailed explanation goes here
    max_val = 256;
    psnr = 20 * log10(max_val / sqrt(mse));

end