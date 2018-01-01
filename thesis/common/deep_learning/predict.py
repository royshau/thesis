#!/home/ohadsh/Tools/anaconda/bin/python
# Predict
"""
This script is using to predict with given snapshot 
"""
import tensorflow as tf
import numpy as np
import argparse
import copy
import os
import sys
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.dl.k_space_graph import *


def predict(data_dir, model, output_dir=None, tt='test', debug=False):

    # Init data base - without shuffle!
    file_names = ['k_space_real', 'k_space_real_gt']
    data_set = KspaceDataSet(data_dir, file_names, stack_size=50, shuffle=False)
    # Create output directories
    out_dir = os.path.join(output_dir, 'predict', tt)
    os.makedirs(out_dir)

    # ckpt = tf.train.get_checkpoint_state(model)

    # Load checkpoint
    saver = tf.train.Saver(tf.all_variables())
    # Start tensorflow interactive session
    sess = tf.InteractiveSession()

    # if ckpt and ckpt.model_checkpoint_path:
    if model:
        # saver.restore(sess, ckpt.model_checkpoint_path)
        saver.restore(sess, model)
        print "Model loaded"
    else:
        print "No checkpoint file found"
    data_set_tt = getattr(data_set, tt)

    predict_counter = 0

    with open(os.path.join(out_dir, 'y_pred.bin'), 'w') as f_pred ,open(os.path.join(out_dir, 'x_upscaled.bin'), 'w') \
            as f_upscaled:
        while data_set_tt.epoch == 0:
            # Running over all data until epoch > 0
            next_batch = copy.deepcopy(data_set_tt.next_batch(FLAGS.mini_batch_size))
            batch_ys = next_batch['k_space_real_gt']
            batch_xs = next_batch['k_space_real']
            feed = {x: batch_xs, y_: batch_ys}
            if len(batch_xs):
                result = sess.run([y_pred, x_image_upscaled], feed_dict=feed)
                f_pred.write(result[0].ravel())
                f_upscaled.write(result[1].ravel())

            predict_counter += FLAGS.mini_batch_size
            print "Done - " + str(predict_counter)


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()

    # parser.add_argument('--data_dir', required=True, type=str, help='Path of data dir')

    # parser.add_argument('--model', required=True, type=str, help='Model path - checkpoint')

    # parser.add_argument('--output_dir', required=False, default=None, type=str, help='Basic output directory for prediction')

    # parser.add_argument('--tt', required=False, type=str, default='test', help='Train or Test data')

    # parser.add_argument('--debug', required=False, default=False, type=bool, help='debug mode')

    # args = parser.parse_args()

    # # Run script:
    # # tf.app.run()
    # predict(args.data_dir, args.model, args.output_dir, args.tt, debug=args.debug)
    tt = 'test'
    data_dir='/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/shuffle/'
    model='/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/2016_02_06/model.ckpt-330000'
    output_dir='/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/runs/2016_02_06/'
    debug=False
    predict(data_dir, model, output_dir, tt, debug=debug)
