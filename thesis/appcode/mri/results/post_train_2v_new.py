# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.utils import get_image_from_kspace, interpolated_missing_samples, zero_padding
from common.files_IO.file_handler import FileHandler
from appcode.mri.k_space.data_creator import get_random_mask, get_subsample, get_random_gaussian_mask, get_rv_mask
from common.viewers.imshow import imshow
file_names = ['k_space_real_gt', 'k_space_imag_gt']
mini_batch = 50
from scipy import ndimage
from collections import defaultdict

start_line = 0    

def post_train_2v(data_dir, predict_paths, h=256, w=256, tt='test', show=False, keep_center=None, DIMS_IN=None, DIMS_OUT=None, sampling_factor=None):
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

    predict_info = {'width': w, 'height': h, 'channels': 1, 'dtype': 'float32'}

    f_predict = defaultdict(dict)
    for (pred_name, pred_path) in predict_paths.iteritems():
        if pred_name != 'interp':
            f_predict[pred_name]['real'] = FileHandler(path=os.path.join(pred_path, "000000.predict_real.bin"),
                                                       info=predict_info, read_or_write='read', name=pred_name)
            f_predict[pred_name]['imag'] = FileHandler(path=os.path.join(pred_path, "000000.predict_imag.bin"),
                                                       info=predict_info, read_or_write='read', name=pred_name)

    data_set = KspaceDataSet(data_dir, file_names, stack_size=50, shuffle=False)

    data_set_tt = getattr(data_set, tt)
    fig, ax = plt.subplots(nrows=2, ncols=4)
    fig.set_size_inches(18.5, 10.5, forward=True)
    
    while data_set_tt.epoch == 0:
        # Running over all data until epoch > 0
        data = data_set_tt.next_batch(mini_batch, norm=False)

        pred_real = {pred_name: pred_io['real'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}
        pred_imag = {pred_name: pred_io['imag'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}

        real_p = {pred_name: pred_data for pred_name, pred_data in pred_real.iteritems()}
        imag_p = {pred_name: pred_data for pred_name, pred_data in pred_imag.iteritems()}

        for i in range(0, data["k_space_real_gt"].shape[0]):

            # Original image
            k_space_real_gt = data["k_space_real_gt"][i,:,:]
            k_space_imag_gt = data["k_space_imag_gt"][i,:,:]
            k_space_amp_gt = np.log(1+np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
            org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

            # Zero Padding
            mask = get_rv_mask(mask_main_dir='/media/ohadsh/Data/ohadsh/work/matlab/thesis/', factor=sampling_factor)
            # mask = get_random_mask(w=256, h=256, factor=sampling_factor, start_line=start_line, keep_center=keep_center)
            reduction = np.sum(mask) / float(mask.ravel().shape[0])
            k_space_real_zero_pad = mask * k_space_real_gt
            k_space_imag_zero_pad = mask * k_space_imag_gt
            k_space_amp_zero_pad = np.log(1+np.sqrt(k_space_real_zero_pad**2 + k_space_imag_zero_pad**2))
            rec_image_zero_pad = get_image_from_kspace(k_space_real_zero_pad,k_space_imag_zero_pad)

            # Network predicted model 1
            name_1 = real_p.keys()[0]
            rec_image_1 = get_image_from_kspace(real_p[name_1], imag_p[name_1])[i,:,:].T
            k_space_amp_predict_1 = np.log(1+np.sqrt(real_p[name_1]**2 + imag_p[name_1]**2))[i,:,:].T

            # Network predicted model 2
            name_2 = real_p.keys()[1]
            rec_image_2 = get_image_from_kspace(real_p[name_2], imag_p[name_2])[i,:,:].T
            k_space_amp_predict_2 = np.log(1+np.sqrt(real_p[name_2]**2 + imag_p[name_2]**2))[i,:,:].T

            ############ Original############
            ax[0][0].set_title('Original Image')
            ax[0][0].imshow(org_image[30:180,30:180], interpolation="none", cmap="gray")

            # ax[1][0].set_title('Original K-space')
            # ax[1][0].imshow(k_space_amp_gt, interpolation="none", cmap="gray")

            ax[1][0].set_title('mask')
            # ax[1][0].imshow(np.log(1+np.abs(org_image - org_image)), interpolation="none", cmap="gray")
            ax[1][0].imshow(mask, interpolation="none", cmap="gray")
            
            ########### Interpolated ############
            ax[0][1].set_title('Rec Image zero Padded:')
            ax[0][1].imshow(rec_image_zero_pad[30:180,30:180], interpolation="none", cmap="gray")
            #
            # ax[1][1].set_title('Interp K-space zero Padded:')
            # ax[1][1].imshow(k_space_amp_zero_pad, interpolation="none", cmap="gray")

            ax[1][1].set_title('Diff-imag')
            ax[1][1].imshow(np.log(1+np.abs(rec_image_zero_pad - org_image)), interpolation="none", cmap="gray")

            ########### DNN 1 ############
            ax[0][2].set_title('DNN - ' + name_1)
            ax[0][2].imshow(rec_image_1[30:180,30:180], interpolation="none", cmap="gray")

            # ax[1][2].set_title('DNN K-space _ ' + name_1)
            # ax[1][2].imshow(k_space_amp_predict_1, interpolation="none", cmap="gray")

            ax[1][2].set_title('Diff-imag')
            ax[1][2].imshow(np.log(1+np.abs(rec_image_1 - org_image)), interpolation="none", cmap="gray")

            ########### DNN 2 ############
            ax[0][3].set_title('DNN  - ' + name_2)
            ax[0][3].imshow(rec_image_2[30:180,30:180], interpolation="none", cmap="gray")

            # ax[1][3].set_title('DNN K-space - ' + name_2)
            # ax[1][3].imshow(k_space_amp_predict_2, interpolation="none", cmap="gray")

            ax[1][3].set_title('Diff-imag')
            ax[1][3].imshow(1+np.log(np.abs(rec_image_2 - org_image)), interpolation="none", cmap="gray")

            plt.draw()
            # plt.tight_layout()
            plt.waitforbuttonpress(timeout=-1)

    plt.close()

if __name__ == '__main__':
    data_dir = '/media/ohadsh/Data/ohadsh/work/data/T1/sagittal/'

    keep_center = 0.05
    DIMS_IN = np.array([256, 256, 1])
    DIMS_OUT = np.array([256, 256, 1])
    sampling_factor = 4

    predict = {'random_mask_factor4_single': '/sheard/googleDrive/Master/runs/server/Wgan/random_mask_rv/IXI/random_mask_factor4_single/predict/train/',
               'random_mask_factor4_D1': '/sheard/googleDrive/Master/runs/server//Wgan/random_mask_rv/IXI/random_mask_factor4_D1/predict/train/'
               }

    w = 256
    h = 256
    tt = 'train'
    show = False
    post_train_2v(data_dir, predict, h, w, tt, show, keep_center=keep_center, DIMS_IN=DIMS_IN, DIMS_OUT=DIMS_OUT, sampling_factor=sampling_factor)
