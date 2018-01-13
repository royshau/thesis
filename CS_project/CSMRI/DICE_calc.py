import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as imp
import os
from CS_ops import *
from CS import CS

NII_SUFFIX = '.nii.gz'
DATA_DIR = 'data/'
masks_names = ['mask66','mask33','mask50','mask10','mask_33_row','mask_50_row','mask_66_row']
seg_suffix ='_brain_seg'
classes = np.array([1,2,3],np.uint16)

sub_dirs = os.listdir(DATA_DIR+'input')
for mask_name in masks_names:
    DICE_CS = [np.array([]),np.array([]),np.array([])]
    DICE_ZF = [np.array([]),np.array([]),np.array([])]
    for case_num,case in enumerate(sub_dirs,start =0):
        nii_path = DATA_DIR+'input/'+case+'/'+case+ seg_suffix +NII_SUFFIX
        rec_dir = DATA_DIR+'rec_images/'+case

        CS_path = os.path.join(rec_dir, case + '_CS_'+mask_name + seg_suffix + NII_SUFFIX)
        ZF_path = os.path.join(rec_dir, case + '_ZF_'+mask_name + seg_suffix + NII_SUFFIX)

        nii_org = nib.load(nii_path)
        nii_CS = nib.load(CS_path)
        nii_ZF = nib.load(ZF_path)
        data_org = nii_org.get_data().astype(np.float32)
        data_CS = nii_CS.get_data().astype(np.float32)
        data_ZF = nii_ZF.get_data().astype(np.float32)

        # for i in xrange(data_org.shape[2]):
        for cls in classes:
            for i in xrange(data_org.shape[2]):
                class_org = data_org[:,:,i] == cls
                class_CS = data_CS[:,:,i] == cls
                class_ZF = data_ZF[:,:,i] == cls
                # print(class_CS[:,:,70].sum())
                # print(class_org[:, :, 70].sum())
                # plt.figure()
                # plt.imshow(class_CS[:,:,70])
                # plt.figure()
                # plt.imshow(class_org[:, :, 70])
                # plt.show()
                if (class_org.sum()!=0):
                    DICE_CS[cls-1] = np.append(DICE_CS[cls-1], calc_dice(class_CS,class_org))
                    DICE_ZF[cls-1] = np.append(DICE_ZF[cls-1], calc_dice(class_ZF, class_org))
    WM_CS_mean = np.mean(DICE_CS[2])
    WM_CS_std = np.std(DICE_CS[2])
    WM_ZF_mean = np.mean(DICE_ZF[2])
    WM_ZF_std = np.std(DICE_ZF[2])
    GM_CS_mean = np.mean(DICE_CS[1])
    GM_CS_std = np.std(DICE_CS[1])
    GM_ZF_mean = np.mean(DICE_ZF[1])
    GM_ZF_std = np.std(DICE_ZF[1])
    CSF_CS_mean = np.mean(DICE_CS[0])
    CSF_CS_std = np.std(DICE_CS[0])
    CSF_ZF_mean = np.mean(DICE_ZF[0])
    CSF_ZF_std = np.std(DICE_ZF[0])
    print('Mask: {4} WM Dice score \nMean CS DICE : {0} , STD : {1} \n Mean ZF DICE: {2} , STD : {3}'.format(WM_CS_mean, WM_CS_std,WM_ZF_mean,WM_ZF_std,mask_name))
    print('Mask: {4} GM Dice score \nMean CS DICE : {0} , STD : {1} \n Mean ZF DICE: {2} , STD : {3}'.format(GM_CS_mean, GM_CS_std,GM_ZF_mean,GM_ZF_std,mask_name))
    print('Mask: {4} CSF Dice score \nMean CS DICE : {0} , STD : {1} \n Mean ZF DICE: {2} , STD : {3}'.format(CSF_CS_mean,
                                                                                                             CSF_CS_std,
                                                                                                             CSF_ZF_mean,
                                                                                                             CSF_ZF_std,
                                                                                                             mask_name))