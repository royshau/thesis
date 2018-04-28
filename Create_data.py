from appcode.mri.data.mri_data_base import MriDataBase
from appcode.mri.k_space.data_creator_multiple import DataCreatorMulti
from appcode.mri.k_space.shuffle_data import shuffle_data
import numpy as np
import matplotlib.pyplot as plt
from appcode.mri.k_space.utils import get_image_from_kspace
import os
from appcode.mri.k_space.k_space_data_set import KspaceDataSet

output_dir = 'appcode/mri/data/IXI/T1'
base_dir = 'appcode/mri/data/IXI/T1/base'
shuffle_out = 'appcode/mri/data/IXI/T1/shuffle'
curr_dir = os.getcwd()
output_dir = os.path.join(curr_dir,output_dir)
base_dir = os.path.join(curr_dir,base_dir)
shuffle_out = os.path.join(curr_dir,shuffle_out)
data_source = MriDataBase('IXI_T1')
data_creator = DataCreatorMulti(data_source, output_dir)
data_creator.create_examples()
shuffle_data(base_dir,shuffle_out,'["train", "test"]',123,'IXI_T1')
file_names = {'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt'}

data_set = KspaceDataSet(shuffle_out, file_names.values(), stack_size=50, data_base='IXI_T1')
next_batch = (data_set.train.next_batch(10))
#plt.imshow(get_image_from_kspace(next_batch['k_space_real_gt'][0,1,:,:].transpose(),next_batch['k_space_imag_gt'][0,1,:,:]),cmap='gray')
#plt.show()
