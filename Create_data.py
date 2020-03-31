from appcode.mri.data.mri_data_base import MriDataBase
from appcode.mri.k_space.data_creator_multiple import DataCreatorMulti
from appcode.mri.k_space.shuffle_data import shuffle_data
import numpy as np
import matplotlib.pyplot as plt
from appcode.mri.k_space.utils import get_image_from_kspace
import os
from appcode.mri.k_space.k_space_data_set import KspaceDataSet

output_dir = '/HOME/data/DCE-MRI'
base_dir = '/HOME/data/DCE-MRI/base'
shuffle_out = '/HOME/data/DCE-MRI/'
curr_dir = ''
output_dir = os.path.join(curr_dir,output_dir)
print(output_dir)
base_dir = os.path.join(curr_dir,base_dir)
shuffle_out = os.path.join(curr_dir,shuffle_out)
data_source = MriDataBase('DCE-MRI')
data_creator = DataCreatorMulti(data_source, output_dir,data_base='DCE-MRI',axial_limits=np.arange(5,22))
data_creator.create_examples()
shuffle_data(base_dir,shuffle_out,'["test"]',123,'DCE-MRI')