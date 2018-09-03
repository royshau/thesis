import os

import glob

import random

import tensorflow as tf

import numpy as np

from appcode.mri.k_space.utils import get_image_from_kspace
import matplotlib.pyplot as plt


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# _bytes is used for string/char values

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

tfrecord_filename = 'train.tfrecords'

# Initiating the writer and creating the tfrecords file.

writer = tf.python_io.TFRecordWriter(tfrecord_filename)

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.


real = glob.glob('/HOME/data/base/train/*/*.k_space_real_gt.bin')
imag = glob.glob('/HOME/data/base/train/*/*.k_space_imag_gt.bin')
meta = glob.glob('/HOME/data/base/train/*/*.meta_data.bin')
shuffled_ind = np.random.permutation(len(real))
sum_r = 0
sum_i = 0
sum_r2 = 0
sum_i2 = 0
count=0
for i in shuffled_ind:
    real_img = np.flip(np.fromfile(real[i],np.float32).reshape(3,256,256),1)
    imag_img = np.flip(np.fromfile(imag[i], np.float32).reshape(3,256,256),1)
    meta_img = np.fromfile(meta[i], np.float32)

    feature = { 'real': _bytes_feature(real_img.tostring()),
                  'imag': _bytes_feature(imag_img.tostring()),
                'meta': _bytes_feature(meta_img.tostring())}
    sum_r = np.sum(real_img)/(256*256)
    sum_i = np.sum(imag_img**2)/(256*256)
    sum_r2 = np.sum(real_img)/(256*256)
    sum_i2 = np.sum(imag_img**2)/(256*256)
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Writing the serialized example.

    writer.write(example.SerializeToString())
    count+=1
    print(count)

print(sum_r/count)
print(sum_i/count)
print(sum_r2/count)
print(sum_i2/count)

writer.close()

