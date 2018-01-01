function ret = norm_ifft(x)
%NORM_IFFT Summary of this function goes here
%   Detailed explanation goes here
    norm_factor = sqrt(length(x(:)));
    ret = norm_factor * ifftshift(ifft2(fftshift(x)));

end

