# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.utils import get_image_from_kspace, interpolated_missing_samples, zero_padding
from common.files_IO.file_handler import FileHandler
from appcode.mri.k_space.data_creator import get_random_mask, get_subsample
from common.viewers.imshow import imshow
file_names = ['image_gt', 'k_space_real_gt', 'k_space_imag_gt', 'mask', 'k_space_real', 'k_space_imag']
mini_batch = 50
from scipy import ndimage
from collections import defaultdict

base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
with open(os.path.join(base_dir, "factors.json"), 'r') as f:
    data_factors = json.load(f)

start_line = 0    

def statistics_cnn(data_dir, predict_paths, h=256, w=256, tt='test', show=False, keep_center=None, DIMS_IN=None, DIMS_OUT=None, sampling_factor=None):
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

    method = 'bilinear'
    predict_info = {'width': w, 'height': h, 'channels': 2, 'dtype': 'float32'}
    interp_info = {'width': w, 'height': h, 'channels': 2, 'dtype': 'float32'}

    f_predict = defaultdict(dict)
    # for (pred_name, pred_path) in predict_paths.iteritems():
    #     if pred_name != 'interp':
    #         f_predict[pred_name]['real'] = FileHandler(path=os.path.join(pred_path, "000000.predict_real.bin"),
    #                                                    info=predict_info, read_or_write='read', name=pred_name)
    #         f_predict[pred_name]['imag'] = FileHandler(path=os.path.join(pred_path, "000000.predict_imag.bin"),
    #                                                    info=predict_info, read_or_write='read', name=pred_name)

    mu_r = np.float32(data_factors['mean']['k_space_real'])
    sigma_r = np.sqrt(np.float32(data_factors['variance']['k_space_real']))
    norm_r = lambda x: (x * sigma_r) + mu_r
    mu_i = np.float32(data_factors['mean']['k_space_imag'])
    sigma_i = np.sqrt(np.float32(data_factors['variance']['k_space_imag']))
    norm_i = lambda x: (x * sigma_i) + mu_i

    f_predict = FileHandler(path=predict_paths['old_cnn'], info=predict_info, read_or_write='read', name='predict')
    f_interp_mc = FileHandler(path=predict_paths['interp'], info=interp_info, read_or_write='read', name='interp_mc')

    data_set = KspaceDataSet(data_dir, file_names, stack_size=50, shuffle=False)

    data_set_tt = getattr(data_set, tt)
    
    error_zero_all = []
    error_interp_all = []
    error_proposed_all = []
    num_of_batches = 1
    batches = 0
    # while data_set_tt.epoch == 0:
    while batches < num_of_batches:
        # Running over all data until epoch > 0
        data = data_set_tt.next_batch(mini_batch, norm=False)

        # pred_real = {pred_name: pred_io['real'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}
        # pred_imag = {pred_name: pred_io['imag'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}
        pred = f_predict.read(n=mini_batch, reshaped=True)
        mc_interp = f_interp_mc.read(n=mini_batch, reshaped=True)

        # real_p = norm_r(pred[:,0,:,:])
        # imag_p = norm_i(pred[:,1,:,:])

        real_p = pred[:,0,:,:]
        imag_p = pred[:,1,:,:]

        real_interp = mc_interp[:,0,:,:]
        imag_interp = mc_interp[:,1,:,:]

        elements_in_batch = real_p.shape[0]
        # elements_in_batch = 50

        for i in range(0, elements_in_batch):

            # Original image
            k_space_real_gt = data["k_space_real_gt"][i,:,:]
            k_space_imag_gt = data["k_space_imag_gt"][i,:,:]
            # k_space_amp_gt = np.log(np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
            org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

            # Interpolation
            mask = get_random_mask(w=256, h=256, factor=sampling_factor, start_line=start_line, keep_center=keep_center)
            # reduction = np.sum(mask) / float(mask.ravel().shape[0])
            # print (reduction)
            k_space_real_gt_int = data["k_space_real_gt"][i,:,:] * mask
            k_space_imag_gt_int = data["k_space_imag_gt"][i,:,:] * mask

            k_space_real_gt_zero = data["k_space_real_gt"][i,:,:] * mask
            k_space_imag_gt_zero = data["k_space_imag_gt"][i,:,:] * mask

            for line in range(0,255):
                missing_line = np.all(mask[line, :] == 0)
                if missing_line:
                    k_space_real_gt_int[line, :] = 0.5*(k_space_real_gt_int[line-1, :] + k_space_real_gt_int[line+1, :])
                    k_space_imag_gt_int[line, :] = 0.5*(k_space_imag_gt_int[line-1, :] + k_space_imag_gt_int[line+1, :])
            # k_space_amp_interp = np.log(np.sqrt(k_space_real_gt_int**2 + k_space_imag_gt_int**2))

            rec_image_interp = get_image_from_kspace(k_space_real_gt_int,k_space_imag_gt_int)
            rec_image_zero = get_image_from_kspace(k_space_real_gt_zero,k_space_imag_gt_zero)


            # Network predicted model 1
            rec_image_1 = get_image_from_kspace(real_p, imag_p)[i,:,:].T
            imshow(rec_image_1, block=True)
            # k_space_amp_predict_1 = np.log(np.sqrt(real_p[name_1]**2 + imag_p[name_1]**2))[i,:,:].T

            error_proposed = np.sum((rec_image_1 - org_image)**2)
            error_interp = np.sum((rec_image_interp - org_image)**2)
            error_zero = np.sum((rec_image_zero - org_image)**2)

            error_zero_all.append(error_zero)
            error_interp_all.append(error_interp)
            error_proposed_all.append(error_proposed)

        batches += 1
        print("Done on %d examples " % (mini_batch*batches))

    mse_zero = np.array(error_zero_all).mean()
    mse_interp = np.array(error_interp_all).mean()
    mse_proposed = np.array(error_proposed_all).mean()

    psnr_std_zero = psnr(np.array(error_zero_all)).std()
    psnr_std_interp = psnr(np.array(error_interp_all)).std()
    psnr_std_proposed = psnr(np.array(error_proposed_all)).std()

    print("MSE-ZERO = %f" % mse_zero)
    print("MSE-INTERP = %f" % mse_interp)
    print("MSE-PROPOSED = %f" % mse_proposed)

    print("PSNR-MEAN-ZERO = %f [dB]" % psnr(mse_zero))
    print("PSNR-MEAN-INTERP = %f [dB]" % psnr(mse_interp))
    print("PSNR-MEAN-PROPOSED = %f [dB]" % psnr(mse_proposed))

    print("PSNR-STD-ZERO = %f [dB]" % psnr_std_zero)
    print("PSNR-STD-INTERP = %f [dB]" % psnr_std_interp)
    print("PSNR-STD-PROPOSED = %f [dB]" % psnr_std_proposed)


def psnr(mse):
    max_val = 256
    psnr = 20*np.log10(max_val / np.sqrt(mse))
    return psnr

if __name__ == '__main__':
    data_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'

    keep_center = 0.05
    DIMS_IN = np.array([256, 256, 1])
    DIMS_OUT = np.array([256, 256, 1])
    sampling_factor = 2

    # predict = {'2017_03_09_ver7_005': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/singleNets/2017_03_09_ver7_factor_005/predict/train/',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #            }

    predict = {'old_cnn': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/multi_channel/2017_02_10_good_results/predict/train/000000.predict.bin',
               'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
               }

    # predict = {'old_cnn': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/multi_channel/2017_01_23.bkp/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #            }

    w = 256

    h = 256
    tt = 'train'
    show = False
    statistics_cnn(data_dir, predict, h, w, tt, show, keep_center=keep_center, DIMS_IN=DIMS_IN, DIMS_OUT=DIMS_OUT, sampling_factor=sampling_factor)
