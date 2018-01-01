from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.data_creator import get_random_mask, get_subsample_forced
import copy

base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
file_names = {'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt'}
import random
import numpy as np

keep_center = 0.1
DIMS_IN = np.array([140, 256, 1])
DIMS_OUT = np.array([256, 256, 1])
sampling_factor = 2


def feed_data(data_set, tt='train', batch_size=10):
    if tt == 'train':
        next_batch = copy.deepcopy(data_set.train.next_batch(batch_size))
        t_phase = True
    else:
        t_phase = False
        next_batch = copy.deepcopy(data_set.test.next_batch(batch_size))

    real = next_batch[file_names['y_r']]
    imag = next_batch[file_names['y_i']]

    if len(real) == 0 or len(imag) == 0:
        return None

    start_line = 0 if random.random() > 0.5 else 1

    mask = get_random_mask(w=DIMS_OUT[0], h=DIMS_OUT[1], factor=sampling_factor, start_line=start_line,
                           keep_center=keep_center)

    feed = {'x_real': get_subsample_forced(image=real, mask=mask, force_width=256),
            'x_imag': get_subsample_forced(image=imag, mask=mask, force_width=256),
            'y_real': real[:, :, :, np.newaxis],
            'y_imag': imag[:, :, :, np.newaxis],
            'mask': mask[np.newaxis, :, :, np.newaxis],
            'train_phase': t_phase
            }
    return feed


if __name__ == '__main__':
    print_train = 500
    batch_size = 10
    data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=50)

    for i in range(1, 1000000):
        feed = feed_data(data_set, tt='train', batch_size=batch_size)
        if i % print_train == 0:
            if feed is not None:
                print feed['x_real'].shape[2]
            print("Done %d examples" % (i * batch_size))
