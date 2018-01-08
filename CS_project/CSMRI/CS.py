from CS_ops import *

def CS(kspace,mask,params):

    masked_data = mask_and_fill(kspace, mask, method=None)

    rec_brain = ifft2c(masked_data)
    rec_brain_old = rec_brain
    for i in xrange(params['iters']):
        rec_brain = wavelet_thresh(rec_brain, params['thresh'], None, params['wave'])
        rec_kspace = fft2c(rec_brain) * (1 - mask) + masked_data
        rec_brain = ifft2c(rec_kspace)
        if i%10==0:
            if np.sum(abs(rec_brain-rec_brain_old))<params['tol']:
                return rec_kspace
            rec_brain_old = rec_brain
    return rec_kspace

