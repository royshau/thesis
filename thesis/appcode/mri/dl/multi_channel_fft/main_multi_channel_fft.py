#!/home/ohadsh/Tools/anaconda/bin/python
""" Train to complete k_space lines with factor 2
k_space real + imaginary are scaled with log function
and paired as Multi-Channel input
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.dl.multi_channel_fft.k_space_multi_channel_fft import KSpaceSuperResolutionMC
from common.deep_learning.helpers import *
import copy
import os
import datetime
import argparse
import json
from numpy.fft import fftshift, ifftshift

# k space data set on loca SSD
base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
file_names = {'x_r': 'k_space_real', 'x_i': 'k_space_imag', 'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt', 'mri':'image_gt'}

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 5000000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('regularization_weight', 5e-4, 'L2 Norm regularization weight.')
flags.DEFINE_integer('mini_batch_size', 10, 'Size of mini batch')
flags.DEFINE_integer('mini_batch_predict', 50, 'Size of mini batch for predict')

# flags.DEFINE_integer('print_test', 10000, 'Print test frequency')
# flags.DEFINE_integer('print_train', 1000, 'Print train frequency')
flags.DEFINE_integer('print_test', 100, 'Print test frequency')
flags.DEFINE_integer('print_train', 10, 'Print train frequency')

flags.DEFINE_boolean('to_show', False, 'View data')


DIMS_IN = np.array([128, 256, 2])
DIMS_OUT = np.array([256, 256, 2])
IMG_MRI = np.array([256, 256, 1])


# flags.DEFINE_string('train_dir', args.train_dir,
#                            """Directory where to write event logs """
                           # """and checkpoint.""")
flags.DEFINE_string('train_dir', "",
                           """Directory where to write event logs """
                           """and checkpoint.""")
logfile = open(os.path.join(FLAGS.train_dir, 'results_%s.log' % str(datetime.datetime.now()).replace(' ', '')), 'w')

with open(os.path.join(base_dir, "factors.json"), 'r') as f:
    data_factors = json.load(f)


def feed_data(data_set, x_input, y_input, train_phase, mri, tt='train', batch_size=10):
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
    mu_r = np.float32(data_factors['mean'][file_names['x_r']])
    sigma_r = np.sqrt(np.float32(data_factors['variance'][file_names['x_r']]))
    norm_r = lambda x: (x - mu_r) / sigma_r

    mu_i = np.float32(data_factors['mean'][file_names['x_i']])
    sigma_i = np.sqrt(np.float32(data_factors['variance'][file_names['x_i']]))
    norm_i = lambda x: (x - mu_i) / sigma_i

    # Feed input as multi-channel: [0: real, 1: imaginary]
    feed = {x_input: np.concatenate((norm_r(next_batch[file_names['x_r']][:, :, :, np.newaxis]),
                                     norm_i(next_batch[file_names['x_i']][:, :, :, np.newaxis])), 3),
            y_input: np.concatenate((norm_r(next_batch[file_names['y_r']][:, :, :, np.newaxis]),
                                     norm_i(next_batch[file_names['y_i']][:, :, :, np.newaxis])), 3),
            train_phase: t_phase,
            mri: fftshift(next_batch[file_names['mri']][:,:,:,np.newaxis])
            }
    return feed


def run_evaluation(sess, feed, eval_op, step, summary_op, writer, tt):
    """
    Run evaluation and save checkpoint
    :param sess: tf session
    :param feed: dictionary feed
    :param step: global step
    :param summary_op:
    :param eval_op:
    :param writer:
    :param tt: TRAIN / TEST
    :return:
    """
    result = sess.run([summary_op, eval_op], feed_dict=feed)
    summary_str = result[0]
    acc = result[1]
    writer.add_summary(summary_str, step)
    print('%s:  Time: %s , Accuracy at step %s: %s' % (tt, datetime.datetime.now(), step, acc))
    logfile.writelines('%s: Time: %s , Accuracy at step %s: %s\n' % (tt, datetime.datetime.now(), step, acc))
    logfile.flush()


def save_checkpoint(sess, saver, step):
    """
    Dump checkpoint
    :param sess: tf session
    :param saver: saver op
    :param step: global step
    :return:
    """
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=step)


def load_graph():
    """
    :return:
    """
    # Init inputs as placeholders
    x_input = tf.placeholder(tf.float32, shape=[None] + list(DIMS_IN), name='x_input')
    y_input = tf.placeholder(tf.float32, shape=[None] + list(DIMS_OUT), name='y_input')
    mri = tf.placeholder(tf.float32, shape=[None] + list(IMG_MRI), name='mri')
    train_phase = tf.placeholder(tf.bool, name='phase_train')
    network = KSpaceSuperResolutionMC(input=x_input, labels=y_input, dims_in=DIMS_IN,
                                      dims_out=DIMS_OUT, batch_size=FLAGS.mini_batch_size,
                                      reg_w=FLAGS.regularization_weight, train_phase=train_phase, factors=data_factors,
                                      mri=mri)
    network.build(FLAGS)
    return network


def train_model(mode, checkpoint=None):
    
    # Import data
    data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=50)

    net = load_graph()

    # Create a saver and keep all checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.summary.merge_all()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'train'), sess.graph)
    writer_test = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'test'), sess.graph)

    if mode == 'resume':
        # saver.restore(sess, checkpoint)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
    else:
        sess.run(init)
    # Train the model, and feed in test data and record summaries every 10 steps
    for i in range(FLAGS.max_steps):

        if i % FLAGS.print_test == 0:
            # Record summary data and the accuracy
            feed = feed_data(data_set, net.input, net.labels, net.train_phase, net.mri,
                             tt='test', batch_size=FLAGS.mini_batch_size)

            if len(feed[net.input]):
                run_evaluation(sess, feed, step=i, summary_op=merged, eval_op=net.evaluation, writer=writer_test, tt='TEST')
                save_checkpoint(sess=sess, saver=saver, step=i)

        else:
            # Training
            feed = feed_data(data_set, net.input, net.labels, net.train_phase,net.mri,
                             tt='train', batch_size=FLAGS.mini_batch_size)
            if len(feed[net.input]):
                # _, dbg, loss_value = sess.run([net.train_step, net.debug, net.loss], feed_dict=feed)
                _,loss_value = sess.run([net.train_step, net.loss], feed_dict=feed)
            if i % FLAGS.print_train == 0:
                run_evaluation(sess, feed, step=i, summary_op=merged, eval_op=net.evaluation, writer=writer, tt='TRAIN')
            # import pdb
            # pdb.set_trace()
            # print(dbg[0].mean(), dbg[1].mean())

    logfile.close()


def evaluate_checkpoint(tt='test', checkpoint=None, output_file=None, output_file_interp=None):
    """
    Evaluate model on specific checkpoint
    :param tt: 'train', 'test'
    :param checkpoint: path to checkpoint
    :param output_file: If not None, the output will write to this path,
    :return:
    """
    # Import data
    data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=50, shuffle=False)

    net = load_graph()

    # Create a saver and keep all checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    # saver = tf.train.import_meta_graph('%s.meta' % checkpoint)
    sess = tf.Session()
    # saver.restore(sess, checkpoint)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
    # all_vars = tf.trainable_variables()
    # for v in all_vars:
    #     print(v.name)

    data_set_tt = getattr(data_set, tt)

    all_acc = []
    predict_counter = 0
    if output_file is not None:
        f_out = open(output_file, 'w')
    if output_file_interp is not None:
        f_interp = open(output_file_interp, 'w')

    print("Evaluate Model using checkpoint: %s, data=%s" % (checkpoint, tt))
    while data_set_tt.epoch == 0:
            # Running over all data until epoch > 0
            feed = feed_data(data_set, net.input, net.labels, net.train_phase,net.mri,
                             tt='train', batch_size=FLAGS.mini_batch_size)
            if len(feed[net.input]):
                predict, result, x_interp = sess.run([net.predict_kspace, net.evaluation, net.x_input_upscale], feed_dict=feed)
                all_acc.append(np.array(result))
                print('Time: %s , Accuracy for mini_batch is: %s' % (datetime.datetime.now(), result))
                if output_file is not None:
                    f_out.write(predict.ravel())
                    f_interp.write(x_interp.ravel())

            predict_counter += FLAGS.mini_batch_predict
            print("Done - " + str(predict_counter))
            
            # HACK
            # print("HACK")
            # break

    if output_file is not None:
        f_out.close()
        f_interp.close()
    print("Total accuracy is: %f" % np.array(all_acc).mean())


def main(args):

    if args.mode == 'train' or args.mode == 'resume':
        train_model(args.mode, args.checkpoint)
    elif args.mode == 'evaluate':
        evaluate_checkpoint(tt=args.tt, checkpoint=args.checkpoint, output_file=args.output_file, output_file_interp=args.output_file_interp)
    # elif mode == 'predict':
    #     predict_checkpoint(tt=args.tt, checkpoint=args.checkpoint, args.output_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script for train super-resolution k-space.')
    parser.add_argument('--mode', dest='mode', choices=['train', 'evaluate', 'predict', 'resume'], type=str, help='mode')
    parser.add_argument('--tt', dest='tt', choices=['train', 'test'], type=str, help='train / test')
    parser.add_argument('--train_dir', dest='train_dir', default='', type=str, help='training directory')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, help='checkpoint full path')
    parser.add_argument('--output_file', dest='output_file', default=None, type=str, help='Output file for predict')
    parser.add_argument('--output_file_interp', dest='output_file_interp', default=None, type=str, help='Output file for interpolation output')
    args = parser.parse_args()

    if args.mode == 'evaluate':
        assert args.tt and args.checkpoint, "Must have tt and checkpoint for evaluate"
    # elif args.mode == 'predict':
    #     assert args.tt and args.checkpoint and args.output_dir , "Must have tt, checkpoint and output_dir for predict"
    elif args.mode == 'resume':
        assert args.checkpoint, "Must have checkpoint for resume"

    main(args)
