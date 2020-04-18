#!/usr/bin/python
""" Train to complete k_space lines with factor 2
k_space real + imaginary are scaled with log function
and paired as Multi-Channel input
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.data_creator import get_rv_mask
from appcode.mri.dl.gan.old.k_space_wgan_im_loss_1d import KspaceWgan
from common.deep_learning.helpers import *
import os
import datetime
import argparse
from collections import defaultdict
import shutil
import inspect
import sys

# k space data set on loca SSD
base_dir = '/HOME/data/shuffle'
# print("working on 140 lines images")
# base_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/2017_03_02_10_percent/shuffle/'
# file_names = {'x_r': 'k_space_real', 'x_i': 'k_space_imag', 'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt'}
file_names = {'y_r': 'k_space_real_gt', 'y_i': 'k_space_imag_gt' , 'm_d':'meta_data'}

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('max_steps', 50000000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
# flags.DEFINE_float('regularization_weight', 5e-4, 'L2 Norm regularization weight.')
flags.DEFINE_float('reg_w', 5e-4, 'L2 Norm regularization weight.')
flags.DEFINE_float('reg_b', 5e-3, 'L2 Norm regularization weight.')
# flags.DEFINE_integer('mini_batch_size', 10, 'Size of mini batch')
flags.DEFINE_integer('mini_batch_size', 5, 'Size of mini batch')
flags.DEFINE_integer('mini_batch_predict', 200, 'Size of mini batch for predict')
flags.DEFINE_integer('max_predict', 50000, 'Number of steps to run trainer.')

flags.DEFINE_float('gen_loss_context', 5, 'Generative loss, kspace context weight.')
flags.DEFINE_float('im_loss_context', 5, 'Generative loss, Image context weight.')

# flags.DEFINE_float('gen_loss_adversarial', 1.0, 'Generative loss, adversarial weight.')
flags.DEFINE_float('gen_loss_adversarial', 0.1, 'Generative loss, adversarial weight.')
flags.DEFINE_integer('iters_no_adv', 1, 'Iters with adv_w=0')

flags.DEFINE_integer('print_test', 5000, 'Print test frequency')
flags.DEFINE_integer('print_train', 500, 'Print train frequency')
# flags.DEFINE_integer('print_test', 50, 'Print test frequency')
# flags.DEFINE_integer('print_train', 10, 'Print train frequency')

flags.DEFINE_integer('num_D_updates', 5, 'Discriminator update freq')
flags.DEFINE_integer('random_sampling_factor', 4, 'Random mask sampling factor')

flags.DEFINE_boolean('to_show', False, 'View data')
flags.DEFINE_string('database', 'IXI_T1', "data base name - for file info")
flags.DEFINE_boolean('dump_debug', False, 'wide_debug_tensorboard')
keep_center = 0.05
DIMS_IN = np.array([3, 256, 256])
DIMS_OUT = np.array([1, 256, 256])

# flags.DEFINE_string('train_dir', args.train_dir,
#                            """Directory where to write event logs """
                           # """and checkpoint.""")
flags.DEFINE_string('train_dir', "",
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS(sys.argv, known_only=True)
logfile = open(os.path.join(FLAGS.train_dir, 'results_%s.log' % str(datetime.datetime.now()).replace(' ', '')), 'w')

mask_1 = get_rv_mask(mask_main_dir='/HOME/thesis/matlab/', factor='50r_1')
mask_2 = get_rv_mask(mask_main_dir='/HOME/thesis/matlab/', factor='50r_2')
mask_3 = get_rv_mask(mask_main_dir='/HOME/thesis/matlab/', factor='50r_3')

#Select GPU:
GPU_ID = '3'
print ('GPU USED: ' + GPU_ID)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

def get_case_idx(case_hash, meta_data):
    """ Get case indices given cash hash and meta data memmap
    :param case_hash:
    :param meta_data:
    :return:
    """
    META_KEYS = {'hash': 0, 'slice': 1, 'bit_pix': 2, 'aug': 3, 'norm_factor': 4}
    idx = np.where(meta_data[:, META_KEYS['hash']] == case_hash)[0]
    slice_idx_rel = np.argsort(meta_data[idx, META_KEYS['slice']])
    slice_idx_abs = idx[slice_idx_rel]
    return slice_idx_abs

def feed_data(data_set,i, y_input, train_phase, tt='train', batch_size=10):
    """
    Feed data into dictionary
    :param data_set: data set object
    :param x_input: x input placeholder list
    :param y_input: y input placeholder list
    :param y_input: y input placeholder list
    :param tt: 'train' or 'test
    :param batch_size: number of examples
    :return:
    """
    if tt == 'train':
        next_batch = data_set.train.next_batch(batch_size)
        t_phase = True
    else:
        t_phase = False
        next_batch = data_set.test.next_batch(batch_size)

    real = next_batch[file_names['y_r']]
    imag = next_batch[file_names['y_i']]
    if len(real) == 0 or len(imag) == 0:
        return None

    if i%3 ==0:
        mask_1_feed = mask_1[np.newaxis, :, :, np.newaxis].transpose(0,3,1,2)
        mask_2_feed = mask_2[np.newaxis, :, :, np.newaxis].transpose(0, 3, 1, 2)
        mask_3_feed = mask_3[np.newaxis, :, :, np.newaxis].transpose(0, 3, 1, 2)
    elif i%3 ==1:
        mask_1_feed = mask_3[np.newaxis, :, :, np.newaxis].transpose(0,3,1,2)
        mask_2_feed = mask_1[np.newaxis, :, :, np.newaxis].transpose(0, 3, 1, 2)
        mask_3_feed = mask_2[np.newaxis, :, :, np.newaxis].transpose(0, 3, 1, 2)
    else:
        mask_1_feed = mask_2[np.newaxis, :, :, np.newaxis].transpose(0,3,1,2)
        mask_2_feed = mask_3[np.newaxis, :, :, np.newaxis].transpose(0, 3, 1, 2)
        mask_3_feed = mask_1[np.newaxis, :, :, np.newaxis].transpose(0, 3, 1, 2)

    # start_line = int(10*random.random() - 5)
    # mask_single = get_random_mask(w=DIMS_OUT[2], h=DIMS_OUT[1], factor=sampling_factor, start_line=start_line, keep_center=keep_center)
    feed = {y_input['real']: real,
            y_input['imag']: imag,
            y_input['mask_1']: mask_1_feed,
            y_input['mask_2']: mask_2_feed,
            y_input['mask_3']: mask_3_feed,
            train_phase: 'train' #Force Train
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

    r_g, r_d, loss_d_fake, loss_d_real, loss_d, loss_g, l2_norm, g_loss_no_reg, d_loss_no_reg,g_loss_im = sess.run([m_op_g, m_op_d, net.d_loss_fake, net.d_loss_real,
                                                                   net.d_loss, net.g_loss, net.evaluation, net.g_loss_no_reg, net.d_loss_no_reg,net.im_loss], feed_dict=feed)

    writer['G'].add_summary(r_g, step)
    writer['D'].add_summary(r_d, step)

    print('%s:  Time: %s , Loss at step %s: D: %s, D_no_reg: %s, G: %s, G_no_reg: %s, L2: %s, L2_Im %s' % (tt, datetime.datetime.now(), step, loss_d, d_loss_no_reg, loss_g, g_loss_no_reg, l2_norm,g_loss_im))
    logfile.writelines('%s: Time: %s , Accuracy at step %s: D: %s, G: %s, L2: %s, L2_Im %s\n' % (tt, datetime.datetime.now(), step, loss_d, loss_g, l2_norm,g_loss_im))
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
    y_input = {'real': tf.placeholder(tf.float32, shape=[None, DIMS_IN[0], DIMS_IN[1], DIMS_IN[2]], name='y_input_real'),
               'imag': tf.placeholder(tf.float32, shape=[None, DIMS_IN[0], DIMS_IN[1], DIMS_IN[2]], name='y_input_imag'),
               'mask_1': tf.placeholder(tf.float32, shape=[1, DIMS_OUT[0], DIMS_OUT[1], DIMS_OUT[2]], name='mask_1'),
               'mask_2': tf.placeholder(tf.float32, shape=[1, DIMS_OUT[0], DIMS_OUT[1], DIMS_OUT[2]], name='mask_2'),
               'mask_3': tf.placeholder(tf.float32, shape=[1, DIMS_OUT[0], DIMS_OUT[1], DIMS_OUT[2]], name='mask_3')
               }

    tf.add_to_collection("placeholders", y_input['real'])
    tf.add_to_collection("placeholders", y_input['imag'])
    tf.add_to_collection("placeholders", y_input['mask_1'])

    train_phase = tf.placeholder(tf.bool, name='phase_train')
    adv_loss_w = tf.placeholder(tf.float32, name='adv_loss_w')
    network = KspaceWgan(input=y_input, dims_in=DIMS_IN,
                                      dims_out=DIMS_OUT, FLAGS=FLAGS, train_phase=train_phase, adv_loss_w=adv_loss_w)
    network.build()
    return network


def train_model(mode, checkpoint=None):

    print("Learning_rate = %f" % FLAGS.learning_rate)
    
    with open(os.path.join(FLAGS.train_dir, 'FLAGS.json'), 'w') as f:
        #json.dump(FLAGS.__dict__, f)
	print('Log not dumpped! fix needed!')

    # Import data
    data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=1600, data_base=FLAGS.database,memmap=False)
    #data_set_mem = KspaceDataSet(base_dir, file_names.values(), stack_size=256, data_base=FLAGS.database, memmap=True)
    #data_set_tt = getattr(data_set_mem,'train')

    #meta_data = data_set_tt.files_obj['meta_data'].memmap

    net = load_graph()

    # Create a saver and keep all checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    writer = defaultdict(dict)
    writer['train']['D'] = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'train', 'D'), sess.graph)
    writer['train']['G'] = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'train', 'G'), sess.graph)

    writer['test']['D'] = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'test', 'D'), sess.graph)
    writer['test']['G'] = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, 'test', 'G'), sess.graph)

    if mode == 'resume':
        saver.restore(sess, checkpoint)
        start_iter = int(checkpoint.split('-')[-1])
        # saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
    else:
        sess.run(init)
        start_iter = 0

    tf.train.write_graph(sess.graph_def, FLAGS.train_dir, 'graph.pbtxt', True)

    gen_loss_adversarial = FLAGS.gen_loss_adversarial
    # gen_loss_adversarial = FLAGS.gen_loss_adversarial
    print("Starting with adv loss = %f" % gen_loss_adversarial)
    print("Starting at iteration number: %d " % start_iter)
    k = 1
    # Train the model, and feed in test data and record summaries every 10 steps
    for i in range(start_iter, FLAGS.max_steps):

        try:
            if i % FLAGS.print_test == 0:
                # Record summary data and the accuracy
                feed = feed_data(data_set,i, net.labels, net.train_phase,
                                 tt='test', batch_size=FLAGS.mini_batch_size)
                if feed is not None:
                    feed[net.adv_loss_w] = gen_loss_adversarial
                    run_evaluation(sess, feed, step=i, net=net, writer=writer['test'], tt='TEST')
                    save_checkpoint(sess=sess, saver=saver, step=i)

            else:
                # Training
                # Update D network
                for it in np.arange(FLAGS.num_D_updates):
                    feed = feed_data(data_set,i+it, net.labels, net.train_phase,
                                     tt='train', batch_size=FLAGS.mini_batch_size)
                    if (feed is not None) and (feed[feed.keys()[0]].shape[0] == FLAGS.mini_batch_size):
                        feed[net.adv_loss_w] = gen_loss_adversarial

                        _, d_loss_fake, d_loss_real, d_loss = \
                            sess.run([net.train_op_d, net.d_loss_fake, net.d_loss_real, net.d_loss], feed_dict=feed)
                        _ = sess.run([net.clip_weights])

                # Update G network
                feed = feed_data(data_set,i, net.labels, net.train_phase,
                                 tt='train', batch_size=FLAGS.mini_batch_size)
                if (feed is not None) and (feed[feed.keys()[0]].shape[0] == FLAGS.mini_batch_size):
                    feed[net.adv_loss_w] = gen_loss_adversarial
                    _, g_loss = sess.run([net.train_op_g, net.g_loss], feed_dict=feed)

                if i % FLAGS.print_train == 0:
                    run_evaluation(sess, feed, step=i, net=net, writer=writer['train'], tt='TRAIN')

                # import pdb
                # pdb.set_trace()
                # print(dbg[0].mean(), dbg[1].mean())
        except KeyboardInterrupt:
            raise
        except:
            print('Error in iteration')
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
    data_set = KspaceDataSet(base_dir, file_names.values(), stack_size=50, shuffle=False, data_base=FLAGS.database)

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
    if output_file is None:
        # Create output directories
        output_file = os.path.join(os.path.abspath(os.path.join(checkpoint, os.pardir)), 'predict', tt)
        os.makedirs(output_file)

    if output_file is not None:
        f_out_real = open(os.path.join(output_file, "000000.predict_real.bin"), 'w')
        f_out_imag = open(os.path.join(output_file, "000000.predict_imag.bin"), 'w')

    gen_loss_adversarial = 1.0
    i = 0
    print("Evaluate Model using checkpoint: %s, data=%s" % (checkpoint, tt))
    while data_set_tt.epoch == 0:
            # Running over all data until epoch > 0
            i+=1
            feed = feed_data(data_set,i, net.labels, net.train_phase,
                             tt=tt, batch_size=FLAGS.mini_batch_size)
            if feed is not None:
                feed[net.adv_loss_w] = gen_loss_adversarial
                predict, result = sess.run([net.predict_g, net.evaluation], feed_dict=feed)

                all_acc.append(np.array(result))
                print('Time: %s , Accuracy for mini_batch is: %s' % (datetime.datetime.now(), result))
                if output_file is not None:
                    f_out_real.write(predict['real'].ravel())
                    f_out_imag.write(predict['imag'].ravel())
            else:
                break
            predict_counter += FLAGS.mini_batch_size
            print("Done - " + str(predict_counter))
            if predict_counter >= FLAGS.max_predict:
                break

    if output_file is not None:
        f_out_real.close()
        f_out_imag.close()
    print("Total accuracy is: %f" % np.array(all_acc).mean())


def main(args):

    print('Initializing')
    if args.mode == 'train' or args.mode == 'resume':

        # Copy scripts to training dir
	print('Training') 
        shutil.copy(os.path.abspath(__file__), args.train_dir)
        model_file = inspect.getfile(KspaceWgan)
        model_file = model_file.split('.py')[0]+'.py'
        shutil.copy(model_file, args.train_dir)
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
    parser.add_argument('--print_train', dest='print_train', type=int, help='Print_Train')
    parser.add_argument('--print_test', dest='print_test', type=int, help='Print Test')
    parser.add_argument('--num_D_updates', dest='num_D_updates', type=int, help='num_D_updates')
    parser.add_argument('--gen_loss_adversarial', dest='gen_loss_adversarial', type=float, help='gen_loss_adversarial')
    parser.add_argument('--gen_loss_context', dest='gen_loss_context', type=float, help='gen_loss_context')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate')
    parser.add_argument('--dump_debug', dest='dump_debug', type=bool, help='dump all images')
    parser.add_argument('--max_predict', dest='max_predict', type=int, default=5000,  help='maximum predict examples')
    parser.add_argument('--mini_batch_size', dest='mini_batch_size', type=int, default=5,  help='mini batch size')
    parser.add_argument('--random_sampling_factor', dest='random_sampling_factor', type=int, default=6, help='Random mask sampling factor')
    parser.add_argument('--database', dest='database', type=str, help='data base name - for file info')
    parser.add_argument('--reg_w', dest='reg_w', type=float, default=5e-4, help='regularization w')
    parser.add_argument('--reg_b', dest='reg_b', type=float, default=5e-4, help='regularization b')

    args = parser.parse_args()

    if args.mode == 'evaluate':
        assert args.tt and args.checkpoint, "Must have tt and checkpoint for evaluate"
    # elif args.mode == 'predict':
    #     assert args.tt and args.checkpoint and args.output_dir , "Must have tt, checkpoint and output_dir for predict"
    elif args.mode == 'resume':
         print('training')
	 assert args.checkpoint, "Must have checkpoint for resume"
    main(args)