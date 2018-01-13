import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as imp
import os
from CS_ops import *
from CS import CS

NII_SUFFIX = '.nii.gz'
IMG_SUFFIX = '.png'
DATA_DIR = 'data/'
masks_names = ['mask66','mask33','mask50','mask10','mask_33_row','mask_50_row','mask_66_row']
# masks_names = ['mask_33_rec','mask_33_cir','mask_unif']
seg_suffix ='_brain'
# seg_suffix =''

slice = 70
classes = np.array([1,2,3],np.uint16)

sub_dirs = os.listdir(DATA_DIR+'input')

for case_num,case in enumerate(sub_dirs,start =0):
    nii_path = DATA_DIR + 'input/' + case + '/' + case + seg_suffix + NII_SUFFIX
    rec_dir = DATA_DIR + 'rec_images/' + case
    # rec_dir = DATA_DIR + 'rec_images_non_standard/' + case

    ext_dir = DATA_DIR + 'ext_slices/' + case
    nii_org = nib.load(nii_path)
    data_org = nii_org.get_data()
    org_slice = data_org[:,:,slice]
    org_slice_path = os.path.join(ext_dir, case + '_Org' +seg_suffix+ IMG_SUFFIX)
    if not os.path.exists(ext_dir):
        os.makedirs(ext_dir)
    plt.imsave(org_slice_path, org_slice, cmap=plt.cm.gray)
    for mask_name in masks_names:
        CS_path = os.path.join(rec_dir, case + '_CS_'+mask_name + seg_suffix + NII_SUFFIX)
        ZF_path = os.path.join(rec_dir, case + '_ZF_'+mask_name + seg_suffix + NII_SUFFIX)

        CS_slice_path = os.path.join(ext_dir, case + '_CS_'+mask_name  +seg_suffix+ IMG_SUFFIX)
        ZF_slice_path = os.path.join(ext_dir, case + '_ZF_'+mask_name  +seg_suffix+ IMG_SUFFIX)

        nii_CS = nib.load(CS_path)
        nii_ZF = nib.load(ZF_path)
        data_CS = nii_CS.get_data()
        data_ZF = nii_ZF.get_data()
        CS_slice = data_CS[:,:,slice]
        ZF_slice = data_ZF[:,:,slice]

        plt.imsave(CS_slice_path, CS_slice, cmap=plt.cm.gray)
        plt.imsave(ZF_slice_path, ZF_slice, cmap=plt.cm.gray)