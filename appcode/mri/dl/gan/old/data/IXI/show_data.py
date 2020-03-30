#!/usr/bin/python
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.data_creator import get_random_gaussian_mask, get_rv_mask
from appcode.mri.k_space.utils import get_image_from_kspace
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy.random
import numpy.matlib

base_dir = '/sheard/Ohad/thesis/data/IXI/data_for_train/'
file_names = ['k_space_real_gt', 'k_space_imag_gt', 'meta_data']
mini_batch = 50
tt = 'train'

modality = 'T1'
# data_names = ['axial', 'coronal', 'sagittal']
data_names = ['sagittal_updated']
random_sampling_factor = 4
data_sets = {}
for data_name in data_names:
    data_sets[data_name] = KspaceDataSet(os.path.join(base_dir, modality, data_name, 'shuffle')
                                         , file_names, stack_size=50, shuffle=False, data_base='IXI_%s' % modality)
    data_sets[data_name] = getattr(data_sets[data_name], tt)

fig, ax = plt.subplots(nrows=2, ncols=4)
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
        # mask = get_random_gaussian_mask(im_shape=(256, 256), peak_probability=0.6, std=40.0, keep_center=0.05)
        # mask = get_random_gaussian_mask(im_shape=(256, 256), peak_probability=0.2, std=10.0, keep_center=0.01)
        # mask = get_random_gaussian_mask(im_shape=(256, 256), peak_probability=0.2, std=10.0, keep_center=0.01)
        mask = get_rv_mask(mask_main_dir='/media/ohadsh/Data/ohadsh/work/matlab/thesis/', factor=random_sampling_factor)

        random_vals = numpy.random.uniform(-0.05, 0.05, (256,256))

        reduction = np.sum(mask) / float(mask.ravel().shape[0])
        print reduction

        zero_padding_image = get_image_from_kspace(k_space_real_gt*mask, k_space_imag_gt*mask)

        # print "Min/Max (%f, %f)" % (k_space_real_gt.min(), k_space_real_gt.max())

        mask_random = (1-mask)*random_vals
        k_space_real_gt = (k_space_real_gt * mask) + mask_random
        k_space_imag_gt = (k_space_imag_gt * mask) + mask_random

        random_padding_image = get_image_from_kspace(k_space_real_gt, k_space_imag_gt)

        ax[0][0].set_title('Original Image - %s' % data_name)
        ax[0][0].imshow(org_image, interpolation="none", cmap="gray")
        ax[0][0].axis('off')

        ax[0][1].set_title('Zero padded')
        ax[0][1].imshow(zero_padding_image, interpolation="none", cmap="gray")
        ax[0][1].axis('off')

        ax[0][2].set_title('random padded')
        ax[0][2].imshow(random_padding_image, interpolation="none", cmap="gray")
        ax[0][2].axis('off')

        ax[0][3].set_title('Mask - reduction=%f' % reduction)
        ax[0][3].imshow(mask, interpolation="none", cmap="gray")
        ax[0][3].axis('off')

        ax[1][0].set_title('Original kspace - %s' % data_name)
        org_kspace = np.log(1+np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
        ax[1][0].imshow(org_kspace, interpolation="none", cmap="gray")
        ax[1][0].axis('off')

        ax[1][1].set_title('Masked kspace - %s' % data_name)
        org_kspace = np.log(1+np.sqrt((mask*k_space_real_gt)**2 + (mask*k_space_imag_gt)**2))
        ax[1][1].imshow(org_kspace, interpolation="none", cmap="gray")
        ax[1][1].axis('off')

        # ax[2][3].set_title('Mask - reduction=%f' % reduction)
        # ax[2][3].imshow(mask_random, interpolation="none", cmap="gray")
        # ax[2][3].axis('off')

    plt.draw()
    fig.tight_layout()
    plt.waitforbuttonpress(timeout=-1)