import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as imp
from CS_ops import *
from CS import CS


params = {}
params['thresh'] = 0.0075
params['wave'] = 'coif5'
params['iters'] = 250
params['tol'] = 0.075

nii = nib.load('test.nii.gz')
data = nii.get_data()

masks = imp.loadmat('masks.mat')
mask50 = np.array(masks['mask50'])

ZF_data = np.zeros_like(data).astype(np.float32)
CS_data = np.zeros_like(data).astype(np.float32)

max_data_val = np.max(abs(data))
data = data.astype(np.float32)/max_data_val

for i in xrange(data.shape[2]):
    kspace = fft2c(data[:,:,i])
    masked_kspace = kspace*mask50
    ZF_data[:,:,i] = abs(ifft2c(masked_kspace))
    CS_rec = CS(kspace,mask50,params)
    CS_data[:,:,i] = abs(ifft2c(CS_rec))
    print(i)

ZF_nii = nib.Nifti1Image(ZF_data, nii.affine)
CS_nii = nib.Nifti1Image(CS_data, nii.affine)
ZF_nii.update_header()
CS_nii.update_header()
nib.save(ZF_nii, 'ZF.nii.gz')
nib.save(CS_nii, 'CS.nii.gz')