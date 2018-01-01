#!/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import matplotlib.pyplot as plt
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.utils import get_image_from_kspace, interpolated_missing_samples, zero_padding
from common.files_IO.file_handler import FileHandler
from common.viewers.imshow import imshow
file_names = ['image_gt', 'k_space_real_gt', 'k_space_imag_gt', 'mask', 'k_space_real', 'k_space_imag']
mini_batch = 50


def post_train_super_resolution(data_dir, predict_real, predict_imag, h=256, w=256, tt='test', show=False):
    """
    This function read predictions (dictionary) and compare it to the data
    :param data_dir: data main directory
    :param predict_real: predict path for real values
    :param predict_imag: predict path for imaginary valeus
    :param h: height
    :param w: width
    :param tt: train or test
    :param show: show flag
    :return:
    """
    method = 'bilinear'
    predict_info = {'width': w, 'height': h, 'channels': 1, 'dtype': 'float32'}
    f_predict_real = FileHandler(path=predict_real[0], info=predict_info, read_or_write='read', name='predict_real')
    f_predict_imag = FileHandler(path=predict_imag[0], info=predict_info, read_or_write='read', name='predict_imag')

    f_interp_real = FileHandler(path=predict_real[1], info=predict_info, read_or_write='read', name='interp_real')
    f_interp_imag = FileHandler(path=predict_imag[1], info=predict_info, read_or_write='read', name='interp_imag')

    data_set = KspaceDataSet(data_dir, file_names, stack_size=50, shuffle=False)

    data_set_tt = getattr(data_set, tt)
    fig, ax = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(18.5, 10.5, forward=True)
    
    while data_set_tt.epoch == 0:
        # Running over all data until epoch > 0
        data = data_set_tt.next_batch(mini_batch, norm=False)
        real_p = f_predict_real.read(n=mini_batch, reshaped=True)
        imag_p = f_predict_imag.read(n=mini_batch, reshaped=True)
        
        real_interp = f_interp_real.read(n=mini_batch, reshaped=True)
        imag_interp = f_interp_imag.read(n=mini_batch, reshaped=True)

        for i in range(0, real_p.shape[0]):

            # Original image
            k_space_real_gt = data["k_space_real_gt"][i,:,:]
            k_space_imag_gt = data["k_space_imag_gt"][i,:,:]
            k_space_amp_gt = np.log(1+np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
            org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

            # Interpolation
            # k_space_real_gt_sub = interpolated_missing_samples(data["k_space_real"][i,:,:], dims_out=(w,h), method=method)
            # k_space_imag_gt_sub = interpolated_missing_samples(data["k_space_imag"][i,:,:], dims_out=(w,h), method=method)
            # k_space_amp_interp = np.log(1+np.sqrt(k_space_real_gt_sub**2 + k_space_imag_gt_sub**2))
            # rec_image_interp = get_image_from_kspace(k_space_real_gt_sub, k_space_imag_gt_sub)
            rec_image_interp = get_image_from_kspace(real_interp, imag_interp)[i,:,:].T
            k_space_amp_interp = np.log(1+np.sqrt(real_interp**2 + imag_interp**2))[i,:,:].T


            # Network predicted model
            rec_image = get_image_from_kspace(real_p, imag_p)[i,:,:].T
            k_space_amp_predict = np.log(1+np.sqrt(real_p**2 + imag_p**2))[i,:,:].T

            ############ Original############
            ax[0][0].set_title('Original Image')
            ax[0][0].imshow(org_image, interpolation="none", cmap="gray")

            ax[1][0].set_title('Original K-space')
            ax[1][0].imshow(k_space_amp_gt, interpolation="none", cmap="gray")
            
            ############ Interpolated ############
            ax[0][1].set_title('Rec Image Interp:%s ' % method)
            ax[0][1].imshow(rec_image_interp, interpolation="none", cmap="gray")

            ax[1][1].set_title('Interp K-space:%s ' % method)
            ax[1][1].imshow(k_space_amp_interp, interpolation="none", cmap="gray")

            ############ DNN ############
            ax[0][2].set_title('DNN Reconstructed Image')
            ax[0][2].imshow(rec_image, interpolation="none", cmap="gray")

            ax[1][2].set_title('DNN K-space')
            ax[1][2].imshow(k_space_amp_predict, interpolation="none", cmap="gray")

            plt.draw()

            plt.waitforbuttonpress(timeout=-1)

    plt.close()

if __name__ == '__main__':
    data_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/shuffle/'
    predict_real = ['/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/2016_11_19/000000.predict.bin', 
                    '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/2016_11_19/000000.interp.bin']
    predict_imag = ['/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/2016_11_19_imag/000000.predict.bin',
                    '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/2016_11_19_imag/000000.interp.bin']
    w = 256
    h = 256
    tt = 'train'
    show = False
    post_train_super_resolution(data_dir, predict_real, predict_imag, h, w, tt, show)
