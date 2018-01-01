import numpy as np
import sys
import matplotlib.pyplot as plt


from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from common.viewers.imshow import imshow
from appcode.mri.k_space.utils import get_image_from_kspace, zero_padding

to_show = True

# k space data set
base_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/2017_03_02_10_percent/shuffle/'
# base_dir = '/media/ohadsh/sheard/Ohad/thesis/data/SchizData/SchizReg/train/2017_03_03_low_pass_64/shuffle/'
# base_dir = '/media/ohadsh/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/shuffle/'

file_names = ['k_space_real_gt', 'k_space_imag_gt', 'mask', 'meta_data', 'image_gt']
data_set = KspaceDataSet(base_dir, file_names, stack_size=50)

batch = data_set.train.next_batch(data_set.train.N_MAX)

if to_show:
    fig, ax = plt.subplots(nrows=2, ncols=2)

    for i in range(0, 50):
        ax[0][0].set_title('mask')
        mask = batch["mask"][i,:,:]
        ax[0][0].imshow(mask, interpolation="none", cmap="gray")

        ax[0][1].set_title('Image - ground truth')
        ax[0][1].imshow(batch["image_gt"][i,:,:], interpolation="none", cmap="gray")

        k_space_real = batch["k_space_real_gt"][i,:,:] * mask
        k_space_imag = batch["k_space_imag_gt"][i,:,:] * mask

        # k_space_real_pad, k_space_imag_pad = zero_padding(k_space_real, k_space_imag, mask)

        k_space_amp = np.log(1+np.sqrt(k_space_real**2 + k_space_imag**2))
        # k_space_amp = k_space_real_pad
        rec_image = get_image_from_kspace(k_space_real, k_space_imag)

        ax[1][0].set_title('Log-k_space real')
        ax[1][0].imshow(k_space_amp, interpolation="none", cmap="gray")

        ax[1][1].set_title('Log-k_space imaginary')
        ax[1][1].imshow(rec_image, interpolation="none", cmap="gray")

        plt.draw()
        plt.waitforbuttonpress(timeout=-1)