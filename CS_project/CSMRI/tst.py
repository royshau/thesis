from CS import CS
import nibabel as nib
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.io as imp
from CS_ops import *

params = {}
params['wave'] = 'coif5'
threshes = [0.0075]
iters = [300]
params['tol'] = 0.075
nii = nib.load('test2.nii.gz')
data = nii.get_data()

norm_data = data.astype(np.float32)/np.max(abs(data))

kspace = fft2c(norm_data[:,:,70])
masks = imp.loadmat('masks.mat')
mask50 = np.array(masks['mask33'])

for thresh in threshes:
    params['thresh'] = thresh
    for iter in iters:
        params['iters'] = iter
        rec_kspace = CS(kspace, mask50, params)
        rec_brain = ifft2c(rec_kspace)

        print('PSNR CS = {0} , iters: {1} , thresh: {2}'.format(psnr(np.int16(abs(rec_brain*np.max(abs(data)))),data[:,:,70]),iter,thresh))

plt.figure(1)
plt.imshow(abs(rec_brain))
plt.show()
plt.figure(2)
plt.imshow(abs(data[:,:,70]))
plt.show()

new_data = norm_data
new_data[:,:,70] = abs(rec_brain)
new_nii = nib.Nifti1Image(new_data, nii.affine)
new_nii.update_header()
nib.save(new_nii, 'new.nii.gz')