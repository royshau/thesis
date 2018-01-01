#!/home/ohadsh/Tools/anaconda/bin/python
# Predict
"""
This script is using to predict with given snapshot
"""
import tensorflow as tf
import argparse
import copy
import os
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
import numpy as np
import json

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('mini_batch_size', 10, 'Size of mini batch')

file_names = {'x_r': 'k_space_real', 'x_i': 'k_space_imag', 'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt', 'mask': 'mask_low_pass_64'}
file_names = {'x_r': 'k_space_real', 'x_i': 'k_space_imag', 'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt', 'mask': 'mask'}
base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
# base_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/2017_03_02_10_percent/shuffle/'

with open(os.path.join(base_dir, "factors.json"), 'r') as f:
    data_factors = json.load(f)

FLAGS.data_factors = data_factors


def forward(model, tensor_name='predict', output_dir=None, out_file_name='predict', tt='train'):

    # Init data base - without shuffle!
    data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=FLAGS.mini_batch_size * 5, shuffle=False)

    # Create output directories
    out_dir = os.path.join(output_dir, 'predict', tt)
    os.makedirs(out_dir)

    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph(model+".meta")
    saver.restore(sess, model)
    graph = tf.get_default_graph()

    # Look for the tensor_name
    tensor_names = [n.name for n in graph.as_graph_def().node if tensor_name == n.name]
    if len(tensor_names) > 1:
        print tensor_names
        assert 0, "There is more than one tensor with the tensor_name=%s" % tensor_name
    elif len(tensor_names) < 1:
        assert 0, "No tensor with the tensor_name=%s" % tensor_name

    # Get output tensor
    data_set_tt = getattr(data_set, tt)
    out_tensor = graph.get_tensor_by_name(tensor_names[0]+":0")
    predict_counter = 0
    x_input = graph.get_tensor_by_name('x_input:0')
    y_input = graph.get_tensor_by_name('y_input:0')

    train_phase = graph.get_tensor_by_name('phase_train:0')

    with open(os.path.join(out_dir, "000000.%s.bin" % out_file_name), 'wb') as f_out:
        print("Forward Model using checkpoint: %s, data=%s, tensor_name=%s" % (model, tt, out_tensor.name))
        while data_set_tt.epoch == 0:
                # Running over all data until epoch > 0
                feed = feed_data(data_set, x_input, y_input, train_phase, tt=tt, batch_size=FLAGS.mini_batch_size)
                if len(feed[x_input]):
                    output = sess.run([out_tensor], feed_dict=feed)
                    f_out.write(output[0].ravel())
                    predict_counter += FLAGS.mini_batch_size
                    print "Done - " + str(predict_counter)


def feed_data(data_set, x_input, y_input, train_phase, tt='train', batch_size=10):
    """
    Feed data into dictionary
    :param data_set: data set object
    :param x_input: x input placeholder list
    :param y_input: y input placeholder list
    :param tt: 'train' or 'test
    :param batch_size: number of examples
    :return:
    """
    if tt == 'train':
        next_batch = copy.deepcopy(data_set.train.next_batch(batch_size))
        t_phase = True
    else:
        t_phase = False
        next_batch = copy.deepcopy(data_set.test.next_batch(batch_size))

    # Normalize data
    mu_r = np.float32(data_factors['mean'][file_names['y_r']])
    sigma_r = np.sqrt(np.float32(data_factors['variance'][file_names['y_r']]))
    norm_r = lambda x: (x - mu_r) / sigma_r

    mu_i = np.float32(data_factors['mean'][file_names['y_i']])
    sigma_i = np.sqrt(np.float32(data_factors['variance'][file_names['y_i']]))
    norm_i = lambda x: (x - mu_i) / sigma_i

    real = norm_r(next_batch[file_names['y_r']])
    imag = norm_i(next_batch[file_names['y_i']])
    y_in = np.concatenate((real[:, :, :, np.newaxis], imag[:, :, :, np.newaxis]), 3)

    x_in = np.concatenate(((real * next_batch[file_names['mask']])[:, :, :, np.newaxis],
                           (imag * next_batch[file_names['mask']])[:, :, :, np.newaxis]), 3)

    # Feed input as multi-channel: [0: real, 1: imaginary]
    feed = {x_input: x_in,
            y_input: y_in,
            train_phase: t_phase
            }
    return feed
# def feed_data(data_set, x_input, y_input, train_phase, tt='train', batch_size=10):
#     """
#     Feed data into dictionary
#     :param data_set: data set object
#     :param x_input: x input placeholder list
#     :param y_input: y input placeholder list
#     :param tt: 'train' or 'test
#     :param batch_size: number of examples
#     :return:
#     """
#     if tt == 'train':
#         next_batch = copy.deepcopy(data_set.train.next_batch(batch_size))
#         t_phase = True
#     else:
#         # t_phase = False
#         print "HACK: use batch norm for test"
#         t_phase = True
#         next_batch = copy.deepcopy(data_set.test.next_batch(batch_size))
#
#     # Normalize data
#     mu_r = np.float32(data_factors['mean'][file_names['x_r']])
#     sigma_r = np.sqrt(np.float32(data_factors['variance'][file_names['x_r']]))
#     norm_r = lambda x: (x - mu_r) / sigma_r
#
#     mu_i = np.float32(data_factors['mean'][file_names['x_i']])
#     sigma_i = np.sqrt(np.float32(data_factors['variance'][file_names['x_i']]))
#     norm_i = lambda x: (x - mu_i) / sigma_i
#
#     y_in = np.concatenate((norm_r(next_batch[file_names['y_r']][:, :, :, np.newaxis]),
#                                      norm_i(next_batch[file_names['y_i']][:, :, :, np.newaxis])), 3)
#     # d_in = y_in.copy()
#     # Feed input as multi-channel: [0: real, 1: imaginary]
#     feed = {x_input: np.concatenate((norm_r(next_batch[file_names['x_r']][:, :, :, np.newaxis]),
#                                      norm_i(next_batch[file_names['x_i']][:, :, :, np.newaxis])), 3),
#             y_input: y_in,
#             train_phase: t_phase
#             }
#     return feed

if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    # print("Visible gpus = %s" % os.environ["CUDA_VISIBLE_DEVICES"])
    parser = argparse.ArgumentParser(description='Main script for predict on trained model.')
    parser.add_argument('--tt', dest='tt', choices=['train', 'test'], type=str, help='train / test')
    parser.add_argument('--tensor_name', dest='tensor_name', default='predict', type=str, help='Output tensor name')
    parser.add_argument('--train_dir', dest='train_dir', default='', type=str, help='training directory')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, help='checkpoint full path')
    parser.add_argument('--output_dir', dest='output_dir', default=None, type=str, help='Output file for predict')
    parser.add_argument('--out_file_name', dest='out_file_name', default='predict', type=str, help='Output file name')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = def_out_dir = os.path.dirname(args.checkpoint)

    # with tf.device('/cpu:0'):
    forward(model=args.checkpoint, output_dir=args.output_dir, tensor_name=args.tensor_name,
            out_file_name=args.out_file_name)
