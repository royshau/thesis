import nibabel as nib
from CS import CS
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.io as imp
from CS_ops import *
import copy
import os

# nii_ref = nib.load('test.nii.gz')
# ref_data = nii_ref.get_data()
#
# new_data = copy.deepcopy(ref_data)
# new_data[:,:,100] = np.zeros_like(new_data[:,:,100])
#
# new_nii = nib.Nifti1Image(new_data, nii_ref.affine)
# new_nii.update_header()
# nib.save(new_nii, 'new.nii.gz')

nii = nib.load('new.nii.gz')
data = nii.get_data()
plt.figure(1)
plt.imshow(data[:,:,70])
plt.figure(2)
plt.imshow(data[:,:,71])
plt.show()