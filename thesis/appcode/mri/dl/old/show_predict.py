import numpy as np
import sys, os
import matplotlib.pyplot as plt

from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from common.viewers.imshow import imshow
from common.files_IO.file_handler import FileHandler

to_show = True

# k space data set
data_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/shuffle/'
file_names = ['k_space_real', 'k_space_real_gt', 'mask', 'image_gt']
data_set = KspaceDataSet(data_dir, file_names, stack_size=50, shuffle=False)
predict_dir = '/media/ohadsh/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/2016_02_06/predict/test/'
tt = 'test'
data_set_tt = getattr(data_set, tt)
mini_batch_size = 20

info = {"width": 256, "height": 256, "channels": 1, "dtype": np.float32}
f_pred = FileHandler(os.path.join(predict_dir, 'y_pred.bin'), info, "read", name=None)
f_x_upscaled= FileHandler(os.path.join(predict_dir, 'x_upscaled.bin'), info, "read", name=None)

batch = data_set_tt.next_batch(mini_batch_size)
y_pred = f_pred.read(mini_batch_size, reshaped=True)
x_upscaled = f_x_upscaled.read(mini_batch_size, reshaped=True)

if to_show:
    fig, ax = plt.subplots(nrows=2, ncols=2)

    fig2, ax2 = plt.subplots(nrows=2, ncols=2)

    for i in range(0, mini_batch_size):

        ax[0][0].set_title('k-space - ground truth')
        imshow(np.log(0.1+np.abs(batch["k_space_real_gt"][i,:,:])), ax=ax[0][0], fig=fig)

        ax[0][1].set_title('k-space - subsample')
        imshow(np.log(0.1+np.abs(batch["k_space_real"][i,:,:])), ax=ax[0][1], fig=fig)

        ax[1][0].set_title('k-space - interpolation')
        imshow(np.log(0.1+np.abs(x_upscaled[i,:,:].transpose())), ax=ax[1][0], fig=fig)

        ax[1][1].set_title('k-space - predicted')
        imshow(np.log(0.1+np.abs(y_pred[i,:,:].transpose())), ax=ax[1][1], fig=fig)

        ax2[0][0].set_title('Original Image')
        imshow(batch['image_gt'][i,:,:], ax=ax2[0][0], fig=fig2)
        ax2[0][1].set_title('Sampling Mask')
        imshow(batch['mask'][i,:,:], ax=ax2[0][1], fig=fig2)
        plt.waitforbuttonpress(timeout=-1)
        print "i = " + str(i)