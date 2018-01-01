#!/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.data_creator import get_random_mask, get_subsample
from appcode.mri.k_space.utils import get_image_from_kspace, interpolated_missing_samples, get_dummy_k_space_and_image
from common.files_IO.file_handler import FileHandler
from common.viewers.imshow import imshow
file_names = ['image_gt', 'k_space_real_gt', 'k_space_imag_gt', 'mask', 'k_space_real', 'k_space_imag']
mini_batch = 50
import random
from scipy.interpolate import interp2d
import matplotlib.colors as colors

base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
with open(os.path.join(base_dir, "factors.json"), 'r') as f:
    data_factors = json.load(f)


def post_train_2v(data_dir, h=256, w=256, tt='test', show=False):
    """
    This function read predictions (dictionary) and compare it to the data
    :param data_dir: data main directory
    :param predict_paths: dictionary
    :param h: height
    :param w: width
    :param tt: train or test
    :param show: show flag
    :return:
    """

    mu_r = np.float32(data_factors['mean']['k_space_real'])
    sigma_r = np.sqrt(np.float32(data_factors['variance']['k_space_real']))
    norm_r = lambda x: (x * sigma_r) + mu_r

    mu_i = np.float32(data_factors['mean']['k_space_imag'])
    sigma_i = np.sqrt(np.float32(data_factors['variance']['k_space_imag']))
    norm_i = lambda x: (x * sigma_i) + mu_i


    method = 'bilinear'
    predict_info = {'width': w, 'height': h, 'channels': 2, 'dtype': 'float32'}


    data_set = KspaceDataSet(data_dir, file_names, stack_size=50, shuffle=False)

    data_set_tt = getattr(data_set, tt)
    fig, ax = plt.subplots(nrows=3, ncols=4)
    fig, ax = plt.subplots(nrows=1, ncols=4)
    fig.set_size_inches(18.5, 10.5, forward=True)

    
    while data_set_tt.epoch == 0:
        # Running over all data until epoch > 0
        data = data_set_tt.next_batch(mini_batch, norm=False)

        for i in range(0, data["k_space_real_gt"].shape[0]):
            # for move in [-4, 0 , 4]:
            for move in [0]:
                # Original image
                k_space_real_gt = data["k_space_real_gt"][i,:,:]
                k_space_imag_gt = data["k_space_imag_gt"][i,:,:]
                k_space_amp_gt = np.log(1+np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
                org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)
                mask = np.ones_like(k_space_real_gt)
                ax[0].imshow(org_image, interpolation="none", cmap="gray")
                ax[0].axis('off')
                # ############ Original############
                # ax[0][0].set_title('Original Image')
                # ax[0][0].imshow(org_image, interpolation="none", cmap="gray")

                # ax[1][0].set_title('Original K-space')
                # ax[1][0].imshow(k_space_amp_gt, interpolation="none", cmap="gray")

                # ax[2][0].set_title('Sampling Mask')
                # ax[2][0].imshow(mask, interpolation="none", cmap="gray")

                # ########### Filter #####################################################################
                # mask = np.zeros_like(k_space_real_gt)
                # # mask[110:150, :] = 1
                # # mask[96:160, 96:160] = 1
                x_rand = random.random()
                # y_rand = random.random()
                movi_x = int(10*x_rand - 5)
                # movi_y = int(10*y_rand - 5)
                # mask[96+movi_x:160+movi_x, 96+movi_y:160+movi_y] = 1

                # # mask[range(0,2,256),:] = 1
                # k_space_real_gt = data["k_space_real_gt"][i,:,:] * mask
                # k_space_imag_gt = data["k_space_imag_gt"][i,:,:] * mask

                # k_space_amp_gt = np.log(1+np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
                # org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

                # ax[0][1].set_title('Filter 1')
                # ax[0][1].imshow(org_image, interpolation="none", cmap="gray")
                # #
                # ax[1][1].set_title('K-space')
                # ax[1][1].imshow(k_space_amp_gt, interpolation="none", cmap="gray")

                # ax[2][1].set_title('Sampling Mask')
                # ax[2][1].imshow(mask, interpolation="none", cmap="gray")


                # ########### Filter #####################################################################
                # # mask = np.zeros_like(k_space_real_gt)
                # # mask[120:140, :] = 1

                # mask = get_random_mask(w=256, h=256, factor=5, start_line=movi_x, keep_center=0.2)
                # reduction = np.sum(mask) / float(mask.ravel().shape[0])
                # print (reduction)
                # k_space_real_gt = data["k_space_real_gt"][i,:,:] * mask
                # k_space_imag_gt = data["k_space_imag_gt"][i,:,:] * mask

                # # k_space_real_gt = get_subsample(k_space_real_gt, mask, 1,1,256)
                # # k_space_imag_gt = get_subsample(k_space_imag_gt, mask, 1,1,256)

                # k_space_real_gt *= mask
                # k_space_imag_gt *= mask

                # k_space_amp_gt = np.log(1+np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
                # org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

                # ax[0][2].set_title('Filter 2')
                # ax[0][2].imshow(org_image, interpolation="none", cmap="gray")

                # ax[1][2].set_title('K-space')
                # ax[1][2].imshow(k_space_amp_gt, interpolation="none", cmap="gray")

                # ax[2][2].set_title('Sampling Mask')
                # ax[2][2].imshow(mask, interpolation="none", cmap="gray")

                # ########### Filter #####################################################################
                # mask = np.zeros_like(k_space_real_gt)
                # mask[120:140, :] = 1
                mask = get_random_mask(w=256, h=256, factor=2, start_line=0, keep_center=0.05)
                reduction = np.sum(mask) / float(mask.ravel().shape[0])
                print (reduction)
                k_space_real_gt = data["k_space_real_gt"][i,:,:] * mask
                k_space_imag_gt = data["k_space_imag_gt"][i,:,:] * mask

                # k_space_real_gt = get_subsample(k_space_real_gt, mask, 1,1,256)
                # k_space_imag_gt = get_subsample(k_space_imag_gt, mask, 1,1,256)

                # k_space_real_gt *= mask
                # k_space_imag_gt *= mask

                # for line in range(0,255):
                #     missing_line = np.all(mask[line, :] == 0)
                #     if missing_line:
                #         k_space_real_gt[line, :] = 0.5*(k_space_real_gt[line-1, :] + k_space_real_gt[line+1, :])
                #         k_space_imag_gt[line, :] = 0.5*(k_space_imag_gt[line-1, :] + k_space_imag_gt[line+1, :])


                org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

                # temp = np.zeros_like(org_image)
                # org_image = get_subsample(org_image, mask, 1,1,256)
                # temp[64:192, :] = org_image
                # kspace, _ = get_dummy_k_space_and_image(temp)
                # k_space_amp_gt = np.log(1+np.sqrt(kspace.real**2 + kspace.imag**2))
                # org_image = get_image_from_kspace(kspace.real,kspace.imag)

                k_space_amp_gt = np.log(1+np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
                k_space_amp_gt = np.log(np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))



                # ax[1].set_title('Filter')
                ax[1].imshow(org_image, interpolation="none", cmap="gray")
                ax[1].axis('off')
                # ax[2].set_title('K-space')
                ax[2].imshow(k_space_amp_gt, interpolation="none", cmap="gray")
                ax[2].axis('off')
                # ax[3].set_title('Sampling Mask')
                ax[3].imshow(mask, interpolation="none", cmap="gray")
                ax[3].axis('off')
                plt.draw()


                plt.waitforbuttonpress(timeout=-1)

    plt.close()

if __name__ == '__main__':
    data_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
    h = 256
    w = 256
    tt = 'train'
    show = False
    post_train_2v(data_dir, h, w, tt, show)
