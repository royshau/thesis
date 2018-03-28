from appcode.mri.data.mri_data_base import MriDataBase
from appcode.mri.k_space.data_creator_multiple import DataCreatorMulti
from appcode.mri.k_space.shuffle_data import shuffle_data
import numpy as np
import matplotlib.pyplot as plt

from appcode.mri.k_space.k_space_data_set import KspaceDataSet

output_dir = '/home/roysh/MRI_Project/Code/appcode/mri/data/IXI/T1'
base_dir = '/home/roysh/MRI_Project/Code/appcode/mri/data/IXI/T1/base'
shuffle_out = '/home/roysh/MRI_Project/Code/appcode/mri/data/IXI/T1/shuffle'

# data_source = MriDataBase('IXI_T1')
# data_creator = DataCreatorMulti(data_source, output_dir)
# data_creator.create_examples()
# shuffle_data(base_dir,output_dir,'["train", "test"]',123,'IXI_T1')
data = np.fromfile('/home/roysh/MRI_Project/Code/appcode/mri/data/IXI/T1/shuffle/train/000000.image_gt.bin', dtype=np.float32)
file_names = {'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt'}

data_set = KspaceDataSet(shuffle_out, file_names.values(), stack_size=50, data_base='IXI_T1')
next_batch = (data_set.train.next_batch(10))
print(next_batch['k_space_real_gt'].shape)