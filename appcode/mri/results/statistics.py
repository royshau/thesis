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
from scipy import stats

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
    
    error_zero_all = []
    error_interp_all = []
    error_proposed_all = []
    num_of_batches = 100
    batches = 0
    # while data_set_tt.epoch == 0:
    while batches < num_of_batches:
        # Running over all data until epoch > 0
        data = data_set_tt.next_batch(mini_batch, norm=False)

        pred_real = {pred_name: pred_io['real'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}
        pred_imag = {pred_name: pred_io['imag'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}

        real_p = {pred_name: pred_data for pred_name, pred_data in pred_real.iteritems()}
        imag_p = {pred_name: pred_data for pred_name, pred_data in pred_imag.iteritems()}

        name_1 = real_p.keys()[0]
        elements_in_batch = real_p[name_1].shape[0]
        # elements_in_batch = 50

        for i in range(0, elements_in_batch):

            # Original image
            k_space_real_gt = data["k_space_real_gt"][i,:,:]
            k_space_imag_gt = data["k_space_imag_gt"][i,:,:]
            # k_space_amp_gt = np.log(np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
            org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

            # Interpolation
            # mask = get_random_mask(w=256, h=256, factor=sampling_factor, start_line=start_line, keep_center=keep_center)
            mask = get_rv_mask(mask_main_dir='/media/ohadsh/Data/ohadsh/work/matlab/thesis/', factor=sampling_factor)
            reduction = np.sum(mask) / float(mask.ravel().shape[0])
            # print (reduction)

            k_space_real_gt_zero = data["k_space_real_gt"][i,:,:] * mask
            k_space_imag_gt_zero = data["k_space_imag_gt"][i,:,:] * mask
            rec_image_zero = get_image_from_kspace(k_space_real_gt_zero,k_space_imag_gt_zero)

            # Network predicted model 1
            rec_image_1 = get_image_from_kspace(real_p[name_1], imag_p[name_1])[i,:,:].T
            # k_space_amp_predict_1 = np.log(np.sqrt(real_p[name_1]**2 + imag_p[name_1]**2))[i,:,:].T

            error_proposed = np.sum((rec_image_1 - org_image)**2)
            error_zero = np.sum((rec_image_zero - org_image)**2)

            error_zero_all.append(error_zero)
            error_proposed_all.append(error_proposed)

        batches += 1
        print("Done on %d examples " % (mini_batch*batches))

    mse_zero = np.array(error_zero_all).mean()
    print stats.ttest_1samp(error_zero_all, mse_zero)

    mse_proposed = np.array(error_proposed_all).mean()
    print stats.ttest_1samp(error_proposed_all, mse_proposed)

    psnr_std_zero = psnr(np.array(error_zero_all)).std()
    psnr_std_proposed = psnr(np.array(error_proposed_all)).std()
    print stats.ttest_1samp(psnr(np.array(error_proposed_all)), psnr_std_proposed)

    print("MSE-ZERO = %f" % mse_zero)
    print("MSE-PROPOSED = %f" % mse_proposed)

    print("PSNR-MEAN-ZERO = %f [dB]" % psnr(mse_zero))
    print("PSNR-MEAN-PROPOSED = %f [dB]" % psnr(mse_proposed))

    print("PSNR-STD-ZERO = %f [dB]" % psnr_std_zero)
    print("PSNR-STD-PROPOSED = %f [dB]" % psnr_std_proposed)


def psnr(mse):
    max_val = 256
    psnr = 20*np.log10(max_val / np.sqrt(mse))
    return psnr

if __name__ == '__main__':
    
    data_dir = '/media/ohadsh/Data/ohadsh/work/data/T1/sagittal/'
    keep_center = 0.05
    DIMS_IN = np.array([256, 256, 1])
    DIMS_OUT = np.array([256, 256, 1])
    sampling_factor = 4

    predict = {'random_mask_factor4_single': '/sheard/googleDrive/Master/runs/server/Wgan/random_mask_rv/IXI/random_mask_factor4_single/predict/train/'}
    # predict = {'random_mask_factor4_D1': '/sheard/googleDrive/Master/runs/server/Wgan/random_mask_rv/IXI/random_mask_factor4_D1/predict/train/'}


    w = 256
    h = 256
    tt = 'train'
    show = False
    print predict
    post_train_2v(data_dir, predict, h, w, tt, show, keep_center=keep_center, DIMS_IN=DIMS_IN, DIMS_OUT=DIMS_OUT, sampling_factor=sampling_factor)
