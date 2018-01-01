# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.utils import get_image_from_kspace, interpolated_missing_samples, zero_padding
from common.files_IO.file_handler import FileHandler
from common.viewers.imshow import imshow
file_names = ['image_gt', 'k_space_real_gt', 'k_space_imag_gt', 'mask', 'k_space_real', 'k_space_imag']
mini_batch = 50
from scipy import ndimage

base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
# base_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/2017_03_02_10_percent/shuffle'
with open(os.path.join(base_dir, "factors.json"), 'r') as f:
    data_factors = json.load(f)


def post_train_2v(data_dir, predict_paths, h=256, w=256, tt='test', show=False):
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

    # mu_r = np.float32(data_factors['mean']['k_space_real'])
    # sigma_r = np.sqrt(np.float32(data_factors['variance']['k_space_real']))
    # norm_r = lambda x: (x * sigma_r) + mu_r
    # norm_r = lambda x: (x - mu_r) / sigma_r

    # mu_i = np.float32(data_factors['mean']['k_space_imag'])
    # sigma_i = np.sqrt(np.float32(data_factors['variance']['k_space_imag']))
    # norm_i = lambda x: (x * sigma_i) + mu_i
    # norm_i = lambda x: (x - mu_i) / sigma_i

    method = 'bilinear'
    predict_info = {'width': w, 'height': h, 'channels': 2, 'dtype': 'float32'}

    f_predict = {}
    for (pred_name, pred_path) in predict_paths.iteritems():
        if pred_name != 'interp':
            f_predict[pred_name] = FileHandler(path=pred_path, info=predict_info, read_or_write='read', name=pred_name)

    f_interp_mc = FileHandler(path=predict_paths['interp'], info=predict_info, read_or_write='read', name='interp_mc')

    data_set = KspaceDataSet(data_dir, file_names, stack_size=50, shuffle=False)

    data_set_tt = getattr(data_set, tt)
    fig, ax = plt.subplots(nrows=2, ncols=4)
    fig.set_size_inches(18.5, 10.5, forward=True)
    
    while data_set_tt.epoch == 0:
        # Running over all data until epoch > 0
        data = data_set_tt.next_batch(mini_batch, norm=False)
        pred_p = {pred_name: pred_io.read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}
        mc_interp = f_interp_mc.read(n=mini_batch, reshaped=True)

        # real_p = {pred_name: norm_r(pred_data[:,0,:,:]) for pred_name, pred_data in pred_p.iteritems()}
        # imag_p = {pred_name: norm_r(pred_data[:,1,:,:]) for pred_name, pred_data in pred_p.iteritems()}

        real_p = {pred_name: pred_data[:,0,:,:] for pred_name, pred_data in pred_p.iteritems()}
        imag_p = {pred_name: pred_data[:,1,:,:] for pred_name, pred_data in pred_p.iteritems()}

        # real_interp = norm_r(mc_interp[:,0,:,:])
        # imag_interp = norm_i(mc_interp[:,1,:,:])

        real_interp = mc_interp[:,0,:,:]
        imag_interp = mc_interp[:,1,:,:]

        for i in range(0, real_interp.shape[0]):

            # Original image
            k_space_real_gt = data["k_space_real_gt"][i,:,:]
            k_space_imag_gt = data["k_space_imag_gt"][i,:,:]
            k_space_amp_gt = np.log(1+np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
            org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

            # Interpolation
            # mask = np.zeros_like(data["mask"][i,:,:])
            # mask[100:150,:] = 1.0

            # Apply low pass
            # sigma = 0.5
            # real_interp2 = ndimage.gaussian_filter(real_interp, sigma)
            # imag_interp2 = ndimage.gaussian_filter(imag_interp, sigma)
            # real_interp = real_interp * mask.T
            # imag_interp = imag_interp * mask.T
            rec_image_interp = get_image_from_kspace(real_interp, imag_interp)[i,:,:].T
            k_space_amp_interp = np.log(1+np.sqrt(real_interp**2 + imag_interp**2))[i,:,:].T

            # Network predicted model 1
            name_1 = real_p.keys()[0]
            # sigma = 1.0
            # Apply low pass
            # real_p[name_1] = ndimage.gaussian_filter(real_p[name_1], sigma)
            # imag_p[name_1] = ndimage.gaussian_filter(imag_p[name_1], sigma)

            # real_p[name_1] = real_p[name_1] * mask.T
            # imag_p[name_1] = imag_p[name_1] * mask.T

            rec_image_1 = get_image_from_kspace(real_p[name_1], imag_p[name_1])[i,:,:].T
            k_space_amp_predict_1 = np.log(1+np.sqrt(real_p[name_1]**2 + imag_p[name_1]**2))[i,:,:].T

            # Network predicted model 2
            name_2 = real_p.keys()[1]
            rec_image_2 = get_image_from_kspace(real_p[name_2], imag_p[name_2])[i,:,:].T
            k_space_amp_predict_2 = np.log(1+np.sqrt(real_p[name_2]**2 + imag_p[name_2]**2))[i,:,:].T

            ############ Original############
            ax[0][0].set_title('Original Image')
            ax[0][0].imshow(org_image, interpolation="none", cmap="gray")

            ax[1][0].set_title('Original K-space')
            ax[1][0].imshow(k_space_amp_gt, interpolation="none", cmap="gray")
            
            ########### Interpolated ############
            ax[0][1].set_title('Rec Image Interp:%s ' % method)
            ax[0][1].imshow(rec_image_interp, interpolation="none", cmap="gray")
            #
            ax[1][1].set_title('Interp K-space:%s ' % method)
            ax[1][1].imshow(k_space_amp_interp, interpolation="none", cmap="gray")

            ########### DNN 1 ############
            ax[0][2].set_title('DNN Reconstructed Image - ' + name_1)
            ax[0][2].imshow(rec_image_1, interpolation="none", cmap="gray")

            ax[1][2].set_title('DNN K-space _ ' + name_1)
            ax[1][2].imshow(k_space_amp_predict_1, interpolation="none", cmap="gray")

            ########### DNN 2 ############
            ax[0][3].set_title('DNN Reconstructed Image  - ' + name_2)
            ax[0][3].imshow(rec_image_2, interpolation="none", cmap="gray")

            ax[1][3].set_title('DNN K-space - ' + name_2)
            ax[1][3].imshow(k_space_amp_predict_2, interpolation="none", cmap="gray")

            plt.draw()

            plt.waitforbuttonpress(timeout=-1)

    plt.close()

if __name__ == '__main__':
    data_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
    # predict = {'gan': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.predict.bin',
    #            'gan_24':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_24_fft/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #            }

    # Multi-Channel
    #1.  /media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/multi_channel/2017_02_10_good_results/predict/train/000000.predict.bin
    #2. /media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/multi_channel/2017_01_23.bkp/000000.predict.bin
    # predict = {'mc_02_10': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/multi_channel/2017_02_10_good_results/predict/train/000000.predict.bin',
    #            'mc_01_23':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/multi_channel/2017_01_23.bkp/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #            }

    # GAN:
    # 1. 'gan_resnet': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/AWS_to_check/2017_02_24_resnet/predict/train/000000.predict.bin'
    # 2. 'gan_25_fft':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_25_fft/predict/train/000000.predict.bin',
    # predict = {'gan_resnet': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/AWS_to_check/2017_02_24_resnet/predict/train/000000.predict.bin',
    #            'gan_25_fft':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_25_fft/predict/train/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #            }

    # GAN with interpolation
    # 1. 'interp2':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/2017_02_28_interp2/predict/train/000000.predict.bin',
    # 2. 'interp3':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/2017_02_28_interp3/predict/train/000000.predict.bin',
    # predict = {'interp3':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/2017_02_28_interp3/predict/train/000000.predict.bin',
    #            'interp2':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/2017_02_28_interp2/predict/train/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #            }

    # GAN with interpolation - update from AWS
    # 1. 'interp_resnet':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update/gan/2017_02_28_interp_resnet/predict/train/000000.predict.bin',
    # 2. 'interp2' : '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update/gan/2017_02_28_interp2/predict/train/000000.predict.bin',
    # predict = {'interp_resnet':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update/gan/2017_02_28_interp_resnet/predict/train/000000.predict.bin',
    #            'interp2' : '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update/gan/2017_02_28_interp2/predict/train/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #            }

    # GAN with interpolation - update Update_30_1
    # 1. 'interp':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_1/2017_02_28_interp/predict/train/000000.predict.bin',
    # 2. 'interp2' : '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_1/2017_02_28_interp2/predict/train/000000.predict.bin',
    # 3. 'interp3': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_1/2017_02_28_interp3/predict/train/000000.predict.bin'
    # 4. 'resnet' : '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_1/2017_02_28_interp_resnet/predict/train/000000.predict.bin'
    # predict = {'interp1':'/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_1/2017_02_28_interp/predict/train/000000.predict.bin',
    #            'interp2' : '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_1/2017_02_28_interp2/predict/train/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #            }
    #
    predict = {'interp3': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_1/2017_02_28_interp3/predict/train/000000.predict.bin',
               'resnet' : '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_1/2017_02_28_interp_resnet/predict/train/000000.predict.bin',
               'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
               }

    # Only geneator , NO discriminator  update Update_30_2
    # predict = {'interp_no_disc': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_2/2017_03_01_interp_no_disc/2017_03_01_interp_no_disc/predict/train/000000.predict.bin',
    #            'interp_no_disc_resnet' : '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/AWS/Update_30_2/2017_03_01_interp_no_disc/2017_03_01_interp_no_disc/predict/train/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #            }

    # Zero Padding 128
    # predict = {'local': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/zeroPadding/mc_128/local/predict/train/000000.predict.bin',
    #            'aws': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/zeroPadding/mc_128/2017_03_03_128_resnet/predict/train/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #           }

    # Zero padding 140 (10 percent)
    # predict = {'2channels-interp': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/2017_03_02_to_networks/predict/train/000000.predict.bin',
    #            'zeroPadding140' : '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/zeroPadding/mc_10_percent/2017_03_02_zero_padding/predict/train/000000.predict.bin',
    #            'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #           }

    # Zero padding low pass 64
    # predict = {'lowPass': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/zeroPadding/mc_low_pass_64/2017_03_02/predict/train/000000.predict.bin',
    #             'lowPassRes': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/interp/zeroPadding/mc_low_pass_64/2017_03_03_low_pass_64_resnet/predict/train/000000.predict.bin',
    #             'interp': '/sheard/googleDrive/Master/runs/factor_2_phase/gan/2017_02_21_fft/000000.interp.bin'
    #       }

    w = 256
    h = 256
    tt = 'train'
    show = False
    post_train_2v(data_dir, predict, h, w, tt, show)
