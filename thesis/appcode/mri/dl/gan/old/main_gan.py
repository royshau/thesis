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
from appcode.mri.dl.gan.k_space_gan import KSpaceSuperResolutionGAN
from common.deep_learning.helpers import *
import copy
import os
import datetime
import argparse
import json
from collections import defaultdict

# k space data set on loca SSD
base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
file_names = {'x_r': 'k_space_real', 'x_i': 'k_space_imag', 'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt'}

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 5000000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-7, 'Initial learning rate.')
flags.DEFINE_float('regularization_weight', 5e-4, 'L2 Norm regularization weight.')
flags.DEFINE_integer('mini_batch_size', 10, 'Size of mini batch')
flags.DEFINE_integer('mini_batch_predict', 50, 'Size of mini batch for predict')

flags.DEFINE_float('gen_loss_context', 1.0, 'Generative loss, context weight.')
flags.DEFINE_float('gen_loss_adversarial', 1e-3, 'Generative loss, adversarial weight.')

# flags.DEFINE_integer('print_test', 10000, 'Print test frequency')
# flags.DEFINE_integer('print_train', 1000, 'Print train frequency')
flags.DEFINE_integer('print_test', 1000, 'Print test frequency')
flags.DEFINE_integer('print_train', 100, 'Print train frequency')

flags.DEFINE_boolean('to_show', False, 'View data')


DIMS_IN = np.array([128, 256, 2])
DIMS_OUT = np.array([256, 256, 2])


# flags.DEFINE_string('train_dir', args.train_dir,
#                            """Directory where to write event logs """
                           # """and checkpoint.""")
flags.DEFINE_string('train_dir', "",
                           """Directory where to write event logs """
                           """and checkpoint.""")
logfile = open(os.path.join(FLAGS.train_dir, 'results_%s.log' % str(datetime.datetime.now()).replace(' ', '')), 'w')

with open(os.path.join(base_dir, "factors.json"), 'r') as f:
    data_factors = json.load(f)


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
    mu_r = np.float32(data_factors['mean'][file_names['x_r']])
    sigma_r = np.sqrt(np.float32(data_factors['variance'][file_names['x_r']]))
    norm_r = lambda x: (x - mu_r) / sigma_r

    mu_i = np.float32(data_factors['mean'][file_names['x_i']])
    sigma_i = np.sqrt(np.float32(data_factors['variance'][file_names['x_i']]))
    norm_i = lambda x: (x - mu_i) / sigma_i

    y_in = np.concatenate((norm_r(next_batch[file_names['y_r']][:, :, :, np.newaxis]),
                                     norm_i(next_batch[file_names['y_i']][:, :, :, np.newaxis])), 3)
    # d_in = y_in.copy()
    # Feed input as multi-channel: [0: real, 1: imaginary]
    feed = {x_input: np.concatenate((norm_r(next_batch[file_names['x_r']][:, :, :, np.newaxis]),
                                     norm_i(next_batch[file_names['x_i']][:, :, :, np.newaxis])), 3),
            y_input: y_in,
            train_phase: t_phase
            }
    return feed


def run_evaluation(sess, feed, net, step, writer, tt):
    """

    :param sess:
    :param feed:
    :param net:
    :param step:
    :param writer:
    :param tt:
    :return:
    """
    m_op_g = tf.summary.merge_all(key='G')
    m_op_d = tf.summary.merge_all(key='D')

    r_g, r_d, loss_d_fake, loss_d_real, loss_d, loss_g, l2_norm = sess.run([m_op_g, m_op_d, net.d_loss_fake, net.d_loss_real,
                                                                   net.d_loss, net.g_loss, net.evaluation], feed_dict=feed)
    writer['G'].add_summary(r_g, step)
    writer['D'].add_summary(r_d, step)

    print('%s:  Time: %s , Loss at step %s: D: %s, G: %s, L2: %s' % (tt, datetime.datetime.now(), step, loss_d, loss_g, l2_norm))
    logfile.writelines('%s: Time: %s , Accuracy at step %s: D: %s, G: %s, L2: %s\n' % (tt, datetime.datetime.now(), step, loss_d, loss_g, l2_norm))
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
    # d_input = tf.placeholder(tf.float32, shape=[None] + list(DIMS_OUT), name='d_input')
    train_phase = tf.placeholder(tf.bool, name='phase_train')
    network = KSpaceSuperResolutionGAN(input=x_input, labels=y_input, dims_in=DIMS_IN,
                                      dims_out=DIMS_OUT, batch_size=FLAGS.mini_batch_size,
                                      reg_w=FLAGS.regularization_weight, train_phase=train_phase)
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

    writer = defaultdict(dict)
    writer['train']['D'] = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'train', 'D'), sess.graph)
    writer['train']['G'] = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'train', 'G'), sess.graph)

    writer['test']['D'] = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'test', 'D'), sess.graph)
    writer['test']['G'] = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'test', 'G'), sess.graph)

    if mode == 'resume':
        saver.restore(sess, checkpoint)
        # saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
    else:
        sess.run(init)

    # Train the model, and feed in test data and record summaries every 10 steps
    for i in range(FLAGS.max_steps):

        #
        if i % FLAGS.print_test == 0:
            # Record summary data and the accuracy
            feed = feed_data(data_set, net.input, net.labels, net.train_phase,
                             tt='test', batch_size=FLAGS.mini_batch_size)

            if len(feed[net.input]):
                run_evaluation(sess, feed, step=i, net=net, writer=writer['test'], tt='TEST')
                save_checkpoint(sess=sess, saver=saver, step=i)

        else:
            # Training
            feed = feed_data(data_set, net.input, net.labels, net.train_phase,
                             tt='train', batch_size=FLAGS.mini_batch_size)
            # sess.run([merged], feed_dict=feed)
            if len(feed[net.input]):

                # Update D network
                _, d_loss_fake, d_loss_real, d_loss = \
                    sess.run([net.train_op_d, net.d_loss_fake, net.d_loss_real, net.d_loss], feed_dict=feed)

                # Update G network
                _, g_loss = sess.run([net.train_op_g, net.g_loss], feed_dict=feed)

                # Run g_optim twice to make sure that d_loss does not go to zero
                # (different from paper)
                # _, g_loss = sess.run([net.train_op_g, net.g_loss], feed_dict=feed)

            if i % FLAGS.print_train == 0:
                run_evaluation(sess, feed, step=i, net=net, writer=writer['train'], tt='TRAIN')
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
    saver.restore(sess, checkpoint)
    # saver.restore(sess, tf.train.latest_checkpoint(checkpoint))

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
            feed = feed_data(data_set, net.input, net.labels, net.train_phase,
                             tt='train', batch_size=FLAGS.mini_batch_size)
            if len(feed[net.input]):
                predict, result, x_interp = sess.run([net.predict_g, net.evaluation, net.x_input_upscale], feed_dict=feed)
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
