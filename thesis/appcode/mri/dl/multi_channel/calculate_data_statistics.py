import numpy as np
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
# k space data set on loca SSD
base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
file_names = {'x_r': 'k_space_real', 'x_i': 'k_space_imag', 'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt'}
data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=2000, shuffle=False)

batch_size = 2000
predict_counter = 0
tt = 'train'

x_r_avg = []
x_i_avg = []
y_r_avg = []
y_i_avg = []

x_r_var = []
x_i_var = []
y_r_var = []
y_i_var = []
y_r_min = []
y_i_min = []

data_set_tt = getattr(data_set, tt)
while data_set_tt.epoch == 0:
        next_batch = data_set_tt.next_batch(batch_size)
        x_input = np.concatenate((next_batch[file_names['x_r']][:, :, :, np.newaxis],
                                     next_batch[file_names['x_i']][:, :, :, np.newaxis]), 3),
        y_input = np.concatenate((next_batch[file_names['y_r']][:, :, :, np.newaxis],
                                     next_batch[file_names['y_i']][:, :, :, np.newaxis]), 3),

        x_r_avg.append(x_input[0][:,:,:,0].mean())
        x_i_avg.append(x_input[0][:,:,:,1].mean())
        y_r_avg.append(y_input[0][:,:,:,0].mean())
        y_i_avg.append(y_input[0][:,:,:,1].mean())

        y_r_min.append(y_input[0][:,:,:,0].min())
        y_i_min.append(y_input[0][:,:,:,1].min())

        x_r_var.append(x_input[0][:,:,:,0].var())
        x_i_var.append(x_input[0][:,:,:,1].var())
        y_r_var.append(y_input[0][:,:,:,0].var())
        y_i_var.append(y_input[0][:,:,:,1].var())

        predict_counter += batch_size
        print("Done - " + str(predict_counter))

print "x_real_mean: " + str(np.array(x_r_avg).mean())
print "x_imag_mean: " + str(np.array(x_i_avg).mean())
print "y_real_mean: " + str(np.array(y_r_avg).mean())
print "y_imag_mean: " + str(np.array(y_i_avg).mean())

print "y_real_min: " + str(np.array(y_r_avg).min())
print "y_imag_min: " + str(np.array(y_i_avg).min())

print "x_real_var: " + str(np.array(x_r_var).mean())
print "x_imag_var: " + str(np.array(x_i_var).mean())
print "y_real_var: " + str(np.array(y_r_var).mean())
print "y_imag_var: " + str(np.array(y_i_var).mean())

all_res = [np.array(x_r_avg).mean(), np.array(x_i_avg).mean(), np.array(y_r_avg).mean(), np.array(y_i_avg).mean(),
           np.array(x_r_var).mean(), np.array(x_i_var).mean(), np.array(y_r_var).mean(), np.array(y_i_var).mean()]

print all_res