import os
import numpy as np
import random
import matplotlib.pyplot as plt
from appcode.mri.k_space.utils import get_image_from_kspace
import glob

import tensorflow as tf

filenames = ["/media/rrtammyfs/Projects/2018/MRIGAN/data/DCE_MRI/dce_test.tfrecords"]
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
    real = tf.transpose(real,[1,2,0])
    imag = tf.transpose(imag,[1,2,0])
    return real,imag


def augment(real,imag):
    concat = tf.concat([real,imag],axis=2)
    concat = tf.image.random_flip_left_right(concat)
    concat = tf.image.random_flip_up_down(concat)
    concat = tf.transpose(concat,[2,0,1])
    concat += tf.random.normal(concat.shape,stddev=0.005)
    real,imag = tf.split(concat,2)
    return real,imag

# Parse the record into tensors.
dataset = dataset.map(_parse_)
dataset = dataset.map(augment)
# Shuffle the dataset
tf.set_random_seed(50)
dataset = dataset.shuffle(buffer_size=12)
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

i=0
GPU_ID = '-1'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

with tf.Session() as sess:
    while True:
        try:
            i+=1
            imag, real = np.array(sess.run([ima, rea])).squeeze()
            break
        except tf.errors.OutOfRangeError:
            break

print("stats")
print(max_real)
print(max_imag)
print(min_real)
print(min_imag)
print(sum_r)
print(sum_i)
print(i)
plt.show()
image = get_image_from_kspace(real[2,1,:,:], imag[2,1,:,:])
plt.imshow(image)
plt.show()
image = get_image_from_kspace(real[3,1,:,:], imag[3,1,:,:])
plt.imshow(image)
plt.show()
print(1)