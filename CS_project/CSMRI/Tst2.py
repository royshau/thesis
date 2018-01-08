from CS import CS
import nibabel as nib
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.io as imp
from CS_ops import *

params = {}
params['wave'] = 'coif5'
threshes = [0.02,0.015,0.01]
iters = [150,75,50,25]

nii_CS = nib.load('CS_brain.nii.gz')
nii_ZF = nib.load('ZF_brain.nii.gz')
nii_org = nib.load('test_brain_seg.nii.gz')
data_org = nii_org.get_data()
data_ZF = nii_ZF.get_data()
data_CS = nii_CS.get_data()

for i in xrange(data_org.shape[2]):
    plt.figure(1)
    plt.imshow(data_org[:,:,i+50])
    plt.figure(2)
    plt.imshow(data_CS[:,:,i+50])
    plt.figure(3)
    plt.imshow(data_ZF[:,:,i+50])
    plt.show()