import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as imp
import skimage.measure
import os
from CS_ops import *
from skimage.measure import compare_ssim as ssim


NII_SUFFIX = '.nii.gz'
DATA_DIR = '/media/rrtammyfs/Projects/2018/MRIGAN/predict/lesions_20_pp/'
# DATA_DIR = '/HOME/RecPF_rec/Lesions_20/'
GT_DIR = '/media/rrtammyfs/Projects/2018/MRIGAN/predict/lesions_20_pp'
masks_names = ['_predict']
modality = ['mprage','flair','t2','pd']

sub_dirs = os.listdir(DATA_DIR)
for mask_name in masks_names:
    for mod in modality:
        PSNR_CS = np.array([])
        PSNR_ZF = np.array([])
        ssim_arr = np.array([])
        for case_num, case in enumerate(sub_dirs, start=0):
            if mod not in case:
                continue
            # print('Working on case' + case)
            nii_path = GT_DIR+'/'+case+'/'+case+NII_SUFFIX
            rec_dir = DATA_DIR+case+'/'

            CS_path = os.path.join(rec_dir, case +mask_name + NII_SUFFIX)
            nii_org = nib.load(nii_path)
            nii_CS = nib.load(CS_path)
            data_org = nii_org.get_data().astype(np.float32)
            data_CS = nii_CS.get_data().astype(np.float32)
            max_data_val = 1
            for i in xrange(data_org.shape[2]):
                ssim_arr = np.append(ssim_arr,ssim(data_org[:, :, i], data_CS[:, :, i],
                                         data_range=data_org[:, :, i].max() - data_org[:, :, i].min()))
                PSNR_CS = np.append(PSNR_CS, psnr(data_CS[:,:,i],data_org[:,:,i]))

        PSNR_CS_mean = np.mean(PSNR_CS)
        PSNR_CS_std = np.std(PSNR_CS)
        ssim_mean = np.mean(ssim_arr)
        ssim_std = np.std(ssim_arr)

        print('Modality: {} \n Mean  PSNR : {} , STD : {} \n SSIM : {} , STD : {} '.format(mod,PSNR_CS_mean, PSNR_CS_std,ssim_mean,ssim_std))

