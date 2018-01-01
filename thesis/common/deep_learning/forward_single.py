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
from appcode.mri.k_space.data_creator import get_random_mask, get_subsample_forced
import numpy as np
import json
import random


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('mini_batch_size', 10, 'Size of mini batch')

file_names = {'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt'}
base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'


def forward(model, tensor_name='predict', output_dir=None, out_file_name='predict', tt='train'):

    # Init data base - without shuffle!
    data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=FLAGS.mini_batch_size * 5, shuffle=False)

    # Create output directories
    out_dir = os.path.join(output_dir, 'predict', tt)
    os.makedirs(out_dir)

    sess = tf.Session()
    saver = tf.train.import_meta_graph(model+".meta")
    saver.restore(sess, model)
    graph = tf.get_default_graph()

    # Look for the tensor_name
    # tensor_names = [n.name for n in graph.as_graph_def().node if tensor_name+"_real" == n.name]
    tensor_names = {}
    tensor_names['real'] = tensor_name+"_real"
    tensor_names['imag'] = tensor_name+"_imag"

    # if len(tensor_names) > 1:
    #     print tensor_names
    #     assert 0, "There is more than one tensor with the tensor_name=%s" % tensor_name
    if len(tensor_names) < 1:
        assert 0, "No tensor with the tensor_name=%s" % tensor_name

    # Get output tensor
    data_set_tt = getattr(data_set, tt)

    out_tensor = [graph.get_tensor_by_name(tensor_names['real']+":0"),graph.get_tensor_by_name(tensor_names['imag']+":0")]
    out_tensor = [graph.get_tensor_by_name('y_input_imag_1:0'),graph.get_tensor_by_name('y_input_imag_1:0')]
    import pdb
    pdb.set_trace()
    # out_tensor = [graph.get_tensor_by_name('G_/x_real_upscale:0'),graph.get_tensor_by_name('G_/x_real_upscale_1:0')]
    predict_counter = 0
    x_input = {'real':graph.get_tensor_by_name('x_input_real:0'), 'imag': graph.get_tensor_by_name('x_input_imag:0')}
    y_input = {'real': graph.get_tensor_by_name('y_input_real:0'), 'imag': graph.get_tensor_by_name('y_input_imag:0')}

    try:
        y_input['mask'] = graph.get_tensor_by_name('mask:0')
    except:
        y_input['mask'] = graph.get_tensor_by_name('y_input_imag_1:0')

    train_phase = graph.get_tensor_by_name('phase_train:0')

    f_out_real = open(os.path.join(out_dir, "000000.predict_real.bin"), 'w')
    f_out_imag = open(os.path.join(out_dir, "000000.predict_imag.bin"), 'w')
    print("Forward Model using checkpoint: %s, data=%s, tensor_name=%s" % (model, tt, tensor_name))
    while data_set_tt.epoch == 0:
            # Running over all data until epoch > 0
            feed = feed_data(data_set, x_input, y_input, train_phase, tt=tt, batch_size=FLAGS.mini_batch_size)
            if feed is not None:
                output = sess.run(out_tensor, feed_dict=feed)
                f_out_real.write(output[0].ravel())
                f_out_imag.write(output[1].ravel())
                predict_counter += FLAGS.mini_batch_size
                print "Done - " + str(predict_counter)

    f_out_real.close()
    f_out_imag.close()


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

    keep_center = 0
    DIMS_IN = np.array([128, 256, 1])
    DIMS_OUT = np.array([256, 256, 1])
    sampling_factor = 2

    # keep_center = 0.1
    # DIMS_IN = np.array([140, 256, 1])
    # DIMS_OUT = np.array([256, 256, 1])
    # sampling_factor = 2

    if tt == 'train':
        next_batch = copy.deepcopy(data_set.train.next_batch(batch_size))
        t_phase = True
    else:
        t_phase = False
        next_batch = copy.deepcopy(data_set.test.next_batch(batch_size))

    real = next_batch[file_names['y_r']]
    imag = next_batch[file_names['y_i']]

    if len(real) == 0 or len(imag) == 0:
        return None

    start_line = 0 if random.random() > 0.5 else 1

    mask = get_random_mask(w=DIMS_OUT[0], h=DIMS_OUT[1], factor=sampling_factor, start_line=start_line, keep_center=keep_center)

    feed = {x_input['real']: get_subsample_forced(image=real, mask=mask, force_width=256),
            x_input['imag']: get_subsample_forced(image=imag, mask=mask, force_width=256),
            y_input['real']: real[:,:,:,np.newaxis],
            y_input['imag']: imag[:,:,:,np.newaxis],
            y_input['mask']: mask[np.newaxis, :, :, np.newaxis],
            train_phase: t_phase
           }

    return feed

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


    forward(model=args.checkpoint, output_dir=args.output_dir, tensor_name=args.tensor_name,
            out_file_name=args.out_file_name)
