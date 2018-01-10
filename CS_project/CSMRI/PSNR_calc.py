import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as imp
import os
from CS_ops import *
from CS import CS

NII_SUFFIX = '.nii.gz'
DATA_DIR = 'data/'
masks_names = ['mask10','mask33','mask50','mask66','mask_33_row','mask_50_row','mask_66_row']

sub_dirs = os.listdir(DATA_DIR+'input')
for mask_name in masks_names:
    PSNR_CS = np.array([])
    PSNR_ZF = np.array([])
    for case_num,case in enumerate(sub_dirs,start =0):
        nii_path = DATA_DIR+'input/'+case+'/'+case+NII_SUFFIX
        rec_dir = DATA_DIR+'rec_images/'+case

        CS_path = os.path.join(rec_dir, case + '_CS_'+mask_name + NII_SUFFIX)
        ZF_path = os.path.join(rec_dir, case + '_ZF_'+mask_name + NII_SUFFIX)

        nii_org = nib.load(nii_path)
        nii_CS = nib.load(CS_path)
        nii_ZF = nib.load(ZF_path)
        data_org = nii_org.get_data().astype(np.float32)
        data_CS = nii_CS.get_data().astype(np.float32)
        data_ZF = nii_ZF.get_data().astype(np.float32)
        max_data_val = np.max(abs(data_org))
        data_org = data_org/max_data_val

        for i in xrange(data_org.shape[2]):
            PSNR_CS = np.append(PSNR_CS, psnr(data_CS[:,:,i],data_org[:,:,i]))
            PSNR_ZF = np.append(PSNR_ZF, psnr(data_ZF[:, :, i], data_org[:, :, i]))
    PSNR_CS_mean = np.mean(PSNR_CS)
    PSNR_CS_std = np.std(PSNR_CS)
    PSNR_ZF_mean = np.mean(PSNR_ZF)
    PSNR_ZF_std = np.std(PSNR_ZF)

    print('Mask: {4} \n Mean CS PSNR : {0} , STD : {1} \n Mean ZF PSNR: {2} , STD : {3}'.format(PSNR_CS_mean, PSNR_CS_std,PSNR_ZF_mean,PSNR_ZF_std,mask_name))

