#!/home/ohadsh/Tools/anaconda/bin/python
""" Train to complete k_space lines with factor 2
k_space real + imaginary are scaled with log function
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.dl.k_space_super_resolution import KSpaceSuperResolution
from common.deep_learning.helpers import *
import copy
import os
import datetime
import time
import argparse

# k space data set
base_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/shuffle/'
# file_names = {'x': 'k_space_real', 'y': 'k_space_real_gt'}
file_names = {'x': 'k_space_imag', 'y': 'k_space_imag_gt'}

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 5000000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
flags.DEFINE_integer('mini_batch_size', 10, 'Size of mini batch')
flags.DEFINE_integer('mini_batch_predict', 50, 'Size of mini batch for predict')
flags.DEFINE_integer('print_test', 10000, 'Print test frequency')
flags.DEFINE_integer('print_train', 1000, 'Print train frequency')
flags.DEFINE_boolean('to_show', False, 'View data')

flags.DEFINE_string('train_dir',
                           '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/2016_11_19/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
flags.DEFINE_string('x', 'k_space_real', 'X input')
flags.DEFINE_string('y', 'k_space_real_gt', 'Y input')


# flags.DEFINE_string('train_dir',
#                            '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/2016_11_19_imag/',
#                            """Directory where to write event logs """
#                            """and checkpoint.""")

# flags.DEFINE_string('x', 'k_space_imag', 'X input')
# flags.DEFINE_string('y', 'k_space_imag_gt', 'Y input')


file_names = {'x': FLAGS.x, 'y': FLAGS.y}

DIMS_IN = np.array([128, 256, 1])
DIMS_OUT = np.array([256, 256, 1])
logfile = open(os.path.join(FLAGS.train_dir, 'results_%s.log' % datetime.datetime.now()), 'w')


def feed_data(data_set, x_input, y_input, tt='train', batch_size=10):
    """
    Feed data into dictionary
    :param data_set: data set object
    :param x_input: x input placeholder
    :param y_input: y input placeholder
    :param tt: 'train' or 'test
    :param batch_size: number of examples
    :return:
    """
    if tt == 'train':
        next_batch = copy.deepcopy(data_set.train.next_batch(batch_size))
    else:
        next_batch = copy.deepcopy(data_set.test.next_batch(batch_size))

    feed = {x_input: next_batch[file_names['x']], y_input: next_batch[file_names['y']]}
    return feed


def run_evaluation(sess, feed, eval_op, step, summary_op, writer):
    """
    Run evaluation and save checkpoint
    :param sess: tf session
    :param feed: dictionary feed
    :param step: global step
    :param summary_op:
    :param eval_op:
    :param writer:
    :return:
    """
    result = sess.run([summary_op, eval_op], feed_dict=feed)
    summary_str = result[0]
    acc = result[1]
    writer.add_summary(summary_str, step)
    print('TEST:  Time: %s , Accuracy at step %s: %s' % (datetime.datetime.now(), step, acc))
    logfile.writelines('TEST: Time: %s , Accuracy at step %s: %s\n' % (datetime.datetime.now(), step, acc))
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
    x_input = tf.placeholder(tf.float32, shape=[None, 128, 256], name='x_input')
    y_input = tf.placeholder(tf.float32, shape=[None, 256, 256], name='y_input')
    network = KSpaceSuperResolution(input=x_input, labels=y_input, dims_in=DIMS_IN, dims_out=DIMS_OUT)

    with tf.name_scope('model'):
        model = network.model()

    with tf.name_scope('loss'):
        loss = network.loss(predict=model, labels=y_input)

    with tf.name_scope('train'):
        # Training evaluation
        # Using Adam solver with cross entropy minimize
        train_step = network.training(s_loss=loss, learning_rate=FLAGS.learning_rate)

    with tf.name_scope('evaluation'):
        # Calculate accuracy
        evaluation = network.evaluation(predict=model, labels=y_input)

    x_upscale = network.x_input_upscale
    return x_input, y_input, model, loss, train_step, evaluation, x_upscale


def train_model(mode, checkpoint=None):
    
    # Import data
    data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=50)

    x_input, y_input, model, loss, train_step, evaluation, x_upscale = load_graph()

    # Create a saver and keep all checkpoints
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()

    sess = tf.Session()
    init = tf.initialize_all_variables()
    writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    if mode == 'resume':
        saver.restore(sess, checkpoint)
    else:
        sess.run(init)
    # Train the model, and feed in test data and record summaries every 10 steps
    for i in range(FLAGS.max_steps):

        if i % FLAGS.print_test == 0:
            # Record summary data and the accuracy
            feed = feed_data(data_set, x_input, y_input, tt='test', batch_size=FLAGS.mini_batch_size)

            if len(feed[x_input]):
                run_evaluation(sess, feed, step=i, summary_op=merged, eval_op=evaluation, writer=writer)
                save_checkpoint(sess=sess, saver=saver, step=i)

        else:
            # Training
            feed = feed_data(data_set, x_input, y_input, tt='train', batch_size=FLAGS.mini_batch_size)
            if len(feed[x_input]):
                _, loss_value = sess.run([train_step, loss], feed_dict=feed)
            if i % FLAGS.print_train == 0:
                print('TRAIN: Time: %s , Loss value at step %s: %s' % (datetime.datetime.now(), i, loss_value))
                logfile.writelines('TRAIN: Time: %s , Loss value at step %s: %s\n' % (datetime.datetime.now(), i, loss_value))
                logfile.flush()
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

    x_input, y_input, model, loss, train_step, evaluation, x_upscale = load_graph()

    # Create a saver and keep all checkpoints
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
    # saver = tf.train.import_meta_graph('%s.meta' % checkpoint)
    sess = tf.Session()
    saver.restore(sess, checkpoint)
    # all_vars = tf.trainable_variables()
    # for v in all_vars:
        # print(v.name)

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
            feed = feed_data(data_set, x_input, y_input, tt=tt, batch_size=FLAGS.mini_batch_predict)
            if len(feed[x_input]):
                predict, result, x_interp = sess.run([model, evaluation, x_upscale], feed_dict=feed)
                all_acc.append(np.array(result))
                print('Time: %s , Accuracy for mini_batch is: %s' % (datetime.datetime.now(), result))
                if output_file is not None:
                    f_out.write(predict.ravel())
                    f_interp.write(x_interp.ravel())

            predict_counter += FLAGS.mini_batch_predict
            print("Done - " + str(predict_counter))
            
            # HACH
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
