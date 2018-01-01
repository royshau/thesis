# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

from tutorial.mnist_conv import DeepMnist
IM_W = 28
IM_H = 28

def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    net = DeepMnist(input_=x, labels=y_, FLAGS=FLAGS)
    net.build()

    graph_location = FLAGS.out_dir
    print('Saving graph to: %s' % graph_location)
    writer = {}

    with tf.Session() as sess:

        # Create a saver and keep all checkpoints
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # Merge all the summaries and write them out to /tmp/mnist_logs
        merged = tf.summary.merge_all()

        writer['train'] = tf.summary.FileWriter(os.path.join(FLAGS.out_dir, 'train'), sess.graph)
        writer['test'] = tf.summary.FileWriter(os.path.join(FLAGS.out_dir, 'test'), sess.graph)

        sess.run(tf.global_variables_initializer())

        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_acc, loss, train_out = sess.run([net.evaluation, net.loss, merged],
                                                      feed_dict={x: batch[0], y_: batch[1]})
                writer['train'].add_summary(train_out, i)
                print('step %d, training accuracy=%g, loss=%g' % (i, train_acc, loss))

            net.train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            if i % 1000 == 0:
                batch = mnist.test.next_batch(200)
                test_acc, test_out = sess.run([net.evaluation, merged],
                                              feed_dict={x: batch[0], y_: batch[1]})
                writer['test'].add_summary(test_out, i)
                print('test accuracy %g' % test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')

    parser.add_argument('--out_dir', type=str,
                        default='/tmp/tensorflow/output/',
                        help='Directory for outputs')

    parser.add_argument('--learning_rate', type=float,
                        default=0.0001,
                        help='learning rate')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
