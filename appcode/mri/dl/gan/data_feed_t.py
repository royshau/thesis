from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.utils import  get_image_from_kspace
import matplotlib.pyplot as plt
import numpy as np

base_dir = '/media/rrtammyfs/Projects/2018/MRIGAN/data/lesions/shuffle'
file_names = {'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt', 'm_d': 'meta_data'}

def feed_data(data_set, tt='train', batch_size=5):
    """
    Feed data into dictionary
    :param data_set: data set object
    :param x_input: x input placeholder list
    :param y_input: y input placeholder list
    :param y_input: y input placeholder list
    :param tt: 'train' or 'test
    :param batch_size: number of examples
    :return:
    """
    if tt == 'train':
        next_batch = data_set.train.next_batch(batch_size)
        t_phase = True
    else:
        t_phase = False
        next_batch = data_set.test.next_batch(batch_size)
    real = next_batch[file_names['y_r']]
    print(real.shape)
    imag = next_batch[file_names['y_i']]
    if len(real) == 0 or len(imag) == 0:
        return None
    img = get_image_from_kspace(real[1,1,:,:],imag[1,1,:,:])
    plt.imshow(img)
    plt.figure()
    plt.imshow(real[1,1,:,:])

    plt.show()
    feed = {'real': real,
            'imag': imag,
            }
    return feed

data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=5, shuffle=False, data_base="Lesions-PP")
feed = feed_data(data_set, tt='test')
