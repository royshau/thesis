import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as imp
import os
from CS_ops import *
from CS import CS

NII_SUFFIX = '.nii.gz'
DATA_DIR = 'data/'
masks_names = ['mask_33_rec','mask_33_cir']

params = {}
params['thresh'] = 0.0075
params['wave'] = 'coif5'
params['iters'] = 250
params['tol'] = 0.075
masks = imp.loadmat('masks.mat')
mask_u = imp.loadmat('mask_unif.mat')




sub_dirs = os.listdir(DATA_DIR+'input')

for case in sub_dirs:
    for mask_name in masks_names:
        mask = np.array(masks[mask_name])
        nii_path = DATA_DIR+'input/'+case+'/'+case+NII_SUFFIX
        rec_dir = DATA_DIR+'rec_images_non_standard/'+case

        CS_path = os.path.join(rec_dir, case + '_CS_'+mask_name + NII_SUFFIX)
        ZF_path = os.path.join(rec_dir, case + '_ZF_'+mask_name + NII_SUFFIX)

        if not os.path.exists(rec_dir):
            os.makedirs(rec_dir)
        print('Working on : {0} mask: {1}'.format(nii_path,mask_name))

        nii = nib.load(nii_path)
        data = nii.get_data()
        ZF_data = np.zeros_like(data).astype(np.float32)
        CS_data = np.zeros_like(data).astype(np.float32)

        max_data_val = np.max(abs(data))
        data = data.astype(np.float32) / max_data_val

        for i in xrange(data.shape[2]):
            kspace = fft2c(data[:, :, i])
            masked_kspace = kspace * mask
            ZF_data[:, :, i] = abs(ifft2c(masked_kspace))
            CS_rec = CS(kspace, mask, params)
            CS_data[:, :, i] = abs(ifft2c(CS_rec))
            if i%20==0:
                print('Slice : {0}'.format(i))

        ZF_nii = nib.Nifti1Image(ZF_data, nii.affine)
        CS_nii = nib.Nifti1Image(CS_data, nii.affine)
        ZF_nii.update_header()
        CS_nii.update_header()

        nib.save(ZF_nii, ZF_path)
        nib.save(CS_nii, CS_path)

    mask_name = 'mask_unif'
    mask = np.array(mask_u[mask_name])
    nii_path = DATA_DIR + 'input/' + case + '/' + case + NII_SUFFIX
    rec_dir = DATA_DIR + 'rec_images_non_standard/' + case

    CS_path = os.path.join(rec_dir, case + '_CS_' + mask_name + NII_SUFFIX)
    ZF_path = os.path.join(rec_dir, case + '_ZF_' + mask_name + NII_SUFFIX)

    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)
    print('Working on : {0} mask: {1}'.format(nii_path, mask_name))

    nii = nib.load(nii_path)
    data = nii.get_data()
    ZF_data = np.zeros_like(data).astype(np.float32)
    CS_data = np.zeros_like(data).astype(np.float32)

    max_data_val = np.max(abs(data))
    data = data.astype(np.float32) / max_data_val

    for i in xrange(data.shape[2]):
        kspace = fft2c(data[:, :, i])
        masked_kspace = kspace * mask
        ZF_data[:, :, i] = abs(ifft2c(masked_kspace))
        CS_rec = CS(kspace, mask, params)
        CS_data[:, :, i] = abs(ifft2c(CS_rec))
        if i % 20 == 0:
            print('Slice : {0}'.format(i))

    ZF_nii = nib.Nifti1Image(ZF_data, nii.affine)
    CS_nii = nib.Nifti1Image(CS_data, nii.affine)
    ZF_nii.update_header()
    CS_nii.update_header()

    nib.save(ZF_nii, ZF_path)
    nib.save(CS_nii, CS_path)