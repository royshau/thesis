#!/usr/bin/python
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.data_creator import get_random_gaussian_mask
from appcode.mri.k_space.utils import get_image_from_kspace
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = '/sheard/Ohad/thesis/data/ADNI/data_for_train/'
file_names = ['k_space_real_gt', 'k_space_imag_gt']
mini_batch = 50
tt = 'train'

slice_to_show = 'sagittal'
data_names = ['NORMAL', 'MCI', 'AD']
data_sets = {}
for data_name in data_names:
    data_sets[data_name] = KspaceDataSet(os.path.join(base_dir, data_name, slice_to_show, 'shuffle')
                                         , file_names, stack_size=50, shuffle=False, data_base='ADNI')
    data_sets[data_name] = getattr(data_sets[data_name], tt)

fig, ax = plt.subplots(nrows=2, ncols=3)
fig.set_size_inches(18.5, 10.5, forward=True)


for i in range(0, 100):

    for data_name in data_names:

        idx = data_names.index(data_name)
        data = data_sets[data_name].next_batch(mini_batch, norm=False)
        k_space_real_gt = data["k_space_real_gt"][i,:,:]
        k_space_imag_gt = data["k_space_imag_gt"][i,:,:]

        # k_space_real_gt = data["k_space_real_gt"][i,45:205,50:210]
        # k_space_imag_gt = data["k_space_imag_gt"][i,45:205,50:210]

        org_image = get_image_from_kspace(k_space_real_gt, k_space_imag_gt)

        # Generating random mask
        # mask = get_random_gaussian_mask(im_shape=(160, 160), peak_probability=0.7, std=45.0, keep_center=0.05)
        mask = get_random_gaussian_mask(im_shape=(256, 256), peak_probability=0.7, std=45.0, keep_center=0.05)

        reduction = np.sum(mask) / float(mask.ravel().shape[0])
        zero_padding_image = get_image_from_kspace(k_space_real_gt*mask, k_space_imag_gt*mask)

        ax[0][idx].set_title('Original Image - %s' % data_name)
        ax[0][idx].imshow(org_image, interpolation="none", cmap="gray")
        ax[0][idx].axis('off')

        ax[1][idx].set_title('Zero padded')
        ax[1][idx].imshow(zero_padding_image, interpolation="none", cmap="gray")
        ax[1][idx].axis('off')

        # ax[idx][2].set_title('Mask - reduction=%f' % reduction)
        # ax[idx][2].imshow(mask, interpolation="none", cmap="gray")
        # ax[idx][2].axis('off')

    plt.draw()
    fig.tight_layout()
    plt.waitforbuttonpress(timeout=-1)