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

    method = 'bilinear'
    predict_info = {'width': w, 'height': h, 'channels': 1, 'dtype': 'float32'}
    interp_info = {'width': w, 'height': h, 'channels': 2, 'dtype': 'float32'}

    f_predict = defaultdict(dict)
    for (pred_name, pred_path) in predict_paths.iteritems():
        if pred_name != 'interp':
            f_predict[pred_name]['real'] = FileHandler(path=os.path.join(pred_path, "000000.predict_real.bin"),
                                                       info=predict_info, read_or_write='read', name=pred_name)
            f_predict[pred_name]['imag'] = FileHandler(path=os.path.join(pred_path, "000000.predict_imag.bin"),
                                                       info=predict_info, read_or_write='read', name=pred_name)

    f_interp_mc = FileHandler(path=predict_paths['interp'], info=interp_info, read_or_write='read', name='interp_mc')

    data_set = KspaceDataSet(data_dir, file_names, stack_size=50, shuffle=False)

    data_set_tt = getattr(data_set, tt)
    fig, ax = plt.subplots(nrows=3, ncols=3)
    fig.set_size_inches(18.5, 10.5, forward=True)
    
    while data_set_tt.epoch == 0:
        # Running over all data until epoch > 0
        data = data_set_tt.next_batch(mini_batch, norm=False)

        pred_real = {pred_name: pred_io['real'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}
        pred_imag = {pred_name: pred_io['imag'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}

        mc_interp = f_interp_mc.read(n=mini_batch, reshaped=True)

        real_p = {pred_name: pred_data for pred_name, pred_data in pred_real.iteritems()}
        imag_p = {pred_name: pred_data for pred_name, pred_data in pred_imag.iteritems()}

        real_interp = mc_interp[:,0,:,:]
        imag_interp = mc_interp[:,1,:,:]

        for i in range(0, data["k_space_real_gt"].shape[0]):

            # Original image
            k_space_real_gt = data["k_space_real_gt"][i,:,:]
            k_space_imag_gt = data["k_space_imag_gt"][i,:,:]
            k_space_amp_gt = np.log(np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
            org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

            # Interpolation
            mask = get_random_mask(w=256, h=256, factor=sampling_factor, start_line=start_line, keep_center=keep_center)
            reduction = np.sum(mask) / float(mask.ravel().shape[0])
            print (reduction)
            k_space_real_gt_int = data["k_space_real_gt"][i,:,:] * mask
            k_space_imag_gt_int = data["k_space_imag_gt"][i,:,:] * mask

            for line in range(0,255):
                missing_line = np.all(mask[line, :] == 0)
                if missing_line:
                    k_space_real_gt_int[line, :] = 0.5*(k_space_real_gt_int[line-1, :] + k_space_real_gt_int[line+1, :])
                    k_space_imag_gt_int[line, :] = 0.5*(k_space_imag_gt_int[line-1, :] + k_space_imag_gt_int[line+1, :])
            k_space_amp_interp = np.log(np.sqrt(k_space_real_gt_int**2 + k_space_imag_gt_int**2))
            rec_image_interp = get_image_from_kspace(k_space_real_gt_int,k_space_imag_gt_int)


            # Network predicted model 1
            name_1 = real_p.keys()[0]
            # sigma = 0.5
            # Apply low pass
            # real_p[name_1] = ndimage.gaussian_filter(real_p[name_1], sigma)
            # imag_p[name_1] = ndimage.gaussian_filter(imag_p[name_1], sigma)

            # real_p[name_1] = real_p[name_1] * mask.T
            # imag_p[name_1] = imag_p[name_1] * mask.T

            # real_p[name_1] *= 1-mask.T
            # imag_p[name_1] *= 1-mask.T
            rec_image_1 = get_image_from_kspace(real_p[name_1], imag_p[name_1])[i,:,:].T

            # im_filter = np.zeros([256,256])
            # filt_cord_y = 40
            # filt_cord_x = 50
            # im_filter[filt_cord_y:-filt_cord_y, filt_cord_x:-filt_cord_x] = 1
            # rec_image_1 *= im_filter

            k_space_amp_predict_1 = np.log(np.sqrt(real_p[name_1]**2 + imag_p[name_1]**2))[i,:,:].T

            ############ Original############
            ax[0][0].set_title('Original Image')
            ax[0][0].imshow(org_image, interpolation="none", cmap="gray")

            ax[1][0].set_title('Original K-space')
            ax[1][0].imshow(k_space_amp_gt, interpolation="none", cmap="gray")

            ax[2][0].set_title('Diff-imag')
            ax[2][0].imshow(np.log(1+np.abs(org_image - org_image)), interpolation="none", cmap="gray")
            
            ########### Interpolated ############
            ax[0][1].set_title('Rec Image Interp:%s ' % method)
            ax[0][1].imshow(rec_image_interp, interpolation="none", cmap="gray")
            #
            ax[1][1].set_title('Interp K-space:%s ' % method)
            ax[1][1].imshow(k_space_amp_interp, interpolation="none", cmap="gray")

            ax[2][1].set_title('Diff-imag')
            ax[2][1].imshow(np.log(1+np.abs(rec_image_interp - org_image)), interpolation="none", cmap="gray")

            ########### DNN 1 ############
            ax[0][2].set_title('DNN Reconstructed Image - ' + name_1)
            ax[0][2].imshow(rec_image_1, interpolation="none", cmap="gray")

            ax[1][2].set_title('DNN K-space _ ' + name_1)
            ax[1][2].imshow(k_space_amp_predict_1, interpolation="none", cmap="gray")

            ax[2][2].set_title('Diff-imag')
            ax[2][2].imshow(np.log(1+np.abs(rec_image_1 - org_image)), interpolation="none", cmap="gray")

            plt.draw()

            plt.waitforbuttonpress(timeout=-1)

    plt.close()

if __name__ == '__main__':
    data_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'

    keep_center = 0.05
    DIMS_IN = np.array([256, 256, 1])
    DIMS_OUT = np.array([256, 256, 1])
    sampling_factor = 2

    predict = {'2017_03_09_ver7_005': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/singleNets/2017_03_09_ver7_factor_005/predict/train/',
               'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
               }
    

    w = 256
    h = 256
    tt = 'train'
    show = False
    post_train_2v(data_dir, predict, h, w, tt, show, keep_center=keep_center, DIMS_IN=DIMS_IN, DIMS_OUT=DIMS_OUT, sampling_factor=sampling_factor)
