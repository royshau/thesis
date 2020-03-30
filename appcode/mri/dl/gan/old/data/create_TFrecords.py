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

tfrecord_filename = 'test_min.tfrecords'

# Initiating the writer and creating the tfrecords file.

writer = tf.python_io.TFRecordWriter(tfrecord_filename)

# Loading the location of all files - image dataset
# Considering our image dataset has apple or orange
# The images are named as apple01.jpg, apple02.jpg .. , orange01.jpg .. etc.


real = glob.glob('/HOME/data/base/test/*/*.k_space_real_gt.bin')
imag = glob.glob('/HOME/data/base/test/*/*.k_space_imag_gt.bin')
meta = glob.glob('/HOME/data/base/test/*/*.meta_data.bin')
shuffled_ind = np.random.permutation(len(real))
shuffled_ind = shuffled_ind[0:20]
for i in shuffled_ind:
    real_img = np.flip(np.fromfile(real[i],np.float32).reshape(3,256,256),1)
    imag_img = np.flip(np.fromfile(imag[i], np.float32).reshape(3,256,256),1)
    meta_img = np.fromfile(meta[i], np.float32)

    feature = { 'real': _bytes_feature(real_img.tostring()),
                  'imag': _bytes_feature(imag_img.tostring()),
                'meta': _bytes_feature(meta_img.tostring())}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Writing the serialized example.

    writer.write(example.SerializeToString())
    print(i)
writer.close()

