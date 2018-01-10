from CS import CS
import nibabel as nib
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.io as imp
from CS_ops import *
masks = imp.loadmat('mask_unif.mat')
for mask in masks:
    print(mask)