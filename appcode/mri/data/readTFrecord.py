import os
import numpy as np
import random
import matplotlib.pyplot as plt
from appcode.mri.k_space.utils import get_image_from_kspace
import glob

import tensorflow as tf

filenames = ["/HOME/data/train.tfrecords"]
dataset = tf.data.TFRecordDataset(filenames)
def _parse_(serialized_example):
    feature = {'real':tf.FixedLenFeature([],tf.string),
                'imag':tf.FixedLenFeature([],tf.string),
               'meta': tf.FixedLenFeature([], tf.string)}
    example = tf.parse_single_example(serialized_example, feature)
    real = tf.decode_raw(example['real'],tf.float32)
    real = tf.reshape(real,[3,256, 256])
    imag = tf.decode_raw(example['imag'], tf.float32)
    imag = tf.reshape(imag, [3,256, 256])
    return real,imag



# Parse the record into tensors.
dataset = dataset.map(_parse_)
# Shuffle the dataset
tf.set_random_seed(50)
dataset = dataset.shuffle(buffer_size=10)
# Repeat the input indefinitly
dataset = dataset.repeat(1)
# Generate batches
dataset = dataset.batch(16)
# Create a one-shot iterator
iterator = dataset.make_one_shot_iterator()
rea, ima = iterator.get_next()
max_real =-10000
max_imag =-10000
min_real =10000
min_imag =10000
sum_r= 0
sum_i= 0
sum_r2= 0
sum_i2= 0
i=0
GPU_ID = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

with tf.Session() as sess:
    while True:
        try:
            imag, real = np.array(sess.run([ima, rea])).squeeze()
            if (np.max(real)>max_real):
                max_real = np.max(real)
            if (np.max(imag)>max_imag):
                max_imag = np.max(imag)
            if (np.min(real)<min_real):
                min_real = np.min(real)
            if (np.min(imag)<min_imag):
                min_imag = np.min(imag)
            sum_r += np.sum(real[:,1,:,:]) / (256 * 256)
            sum_i += np.sum(imag[:,1,:,:]) / (256 * 256)
            sum_r2 += np.sum(real[:,1,:,:]**2) / (256 * 256)
            sum_i2 += np.sum(imag[:,1,:,:]**2) / (256 * 256)
            i+=16
            print(i)
        except tf.errors.OutOfRangeError:
            break
print("stats")
print(max_real)
print(max_imag)
print(min_real)
print(min_imag)
print(sum_r/i)
print(sum_i/i)
print(sum_r2/i)
print(sum_i2/i)
print(i)
# plt.show()
# image = get_image_from_kspace(real[2,1,:,:], imag[2,1,:,:])
# plt.imshow(image)
# plt.show()
# image = get_image_from_kspace(real[3,1,:,:], imag[3,1,:,:])
# plt.imshow(image)
# plt.show()
# print(1)