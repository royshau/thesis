#!/usr/bin/python
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.data_creator import get_random_gaussian_mask
from appcode.mri.k_space.utils import get_image_from_kspace
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

sys.path.append('/media/ohadsh/Data/ohadsh/work/python/bart-0.4.01/python')
import cfl

base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
file_names = ['k_space_real_gt', 'k_space_imag_gt']
mini_batch = 50
tt = 'train'
data_set = KspaceDataSet(base_dir, file_names, stack_size=50, shuffle=False)
data_set_tt = getattr(data_set, tt)

data = data_set_tt.next_batch(mini_batch, norm=False)

k_space_real_gt = data["k_space_real_gt"][0,:,:]
k_space_imag_gt = data["k_space_imag_gt"][0,:,:]
k_space_tuple = (np.zeros_like(k_space_real_gt) + 0j).astype('complex64')
k_space_tuple.real = k_space_real_gt
k_space_tuple.imag = k_space_imag_gt
cfl.writecfl('example', k_space_tuple);
# A = cfl.readcfl('xyz');


# for i in range(0,100):

#     data = data_set_tt.next_batch(mini_batch, norm=False)
#     k_space_real_gt = data["k_space_real_gt"][i,:,:]
#     k_space_imag_gt = data["k_space_imag_gt"][i,:,:]
#     org_image = get_image_from_kspace(k_space_real_gt, k_space_imag_gt)

#     # Generating random mask
#     mask = get_random_gaussian_mask(im_shape=(256, 256), peak_probability=0.7, std=45.0, keep_center=0.05)
    
    
#     reduction = np.sum(mask) / float(mask.ravel().shape[0])
#     zero_padding_image = get_image_from_kspace(k_space_real_gt*mask, k_space_imag_gt*mask)

#     ax[0].set_title('Original Image')
#     ax[0].imshow(org_image, interpolation="none", cmap="gray")
#     ax[0].axis('off')

#     ax[1].set_title('Zero padded')
#     ax[1].imshow(zero_padding_image, interpolation="none", cmap="gray")
#     ax[1].axis('off')

#     ax[2].set_title('Mask - reduction=%f' % reduction)
#     ax[2].imshow(mask, interpolation="none", cmap="gray")
#     ax[2].axis('off')


