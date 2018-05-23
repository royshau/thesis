import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as imp
import os
from CS_ops import *


NII_SUFFIX = '.nii.gz'
DATA_DIR = '/HOME/Trains/22_5_multimask/predict/test/output/'
masks_names = ['']

sub_dirs = os.listdir(DATA_DIR)
for mask_name in masks_names:
    PSNR_CS = np.array([])
    PSNR_ZF = np.array([])
    for case_num,case in enumerate(sub_dirs,start =0):
        print('Working on case' + case)
        nii_path = DATA_DIR+'/'+case+'/'+case+NII_SUFFIX
        rec_dir = DATA_DIR+case+'/'

        CS_path = os.path.join(rec_dir, case + '_predict'+mask_name + NII_SUFFIX)

        nii_org = nib.load(nii_path)
        nii_CS = nib.load(CS_path)
        data_org = nii_org.get_data().astype(np.float32)
        data_CS = nii_CS.get_data().astype(np.float32)
        max_data_val = np.max(abs(data_org))
        data_org = data_org/max_data_val

        for i in xrange(data_org.shape[2]):
            PSNR_CS = np.append(PSNR_CS, psnr(data_CS[:,:,i],data_org[:,:,i]))
    PSNR_CS_mean = np.mean(PSNR_CS)
    PSNR_CS_std = np.std(PSNR_CS)

    print('Mask: {2} \n Mean CS PSNR : {0} , STD : {1} \n '.format(PSNR_CS_mean, PSNR_CS_std,mask_name))

