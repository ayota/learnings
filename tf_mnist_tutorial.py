"""
Convert this to mnist_tf_tutorial class-based model format.
"""
import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils

# x = image
# y_ = label

class MNISTModel(object):
    """
    Predicting numbers depicted in MNIST data set using conv neural nets.

    Structure:
    - Two pooling layers
    - One fully connected layer
    - Readout layer

    """

    def __init__(self, image, label, keep_prob):
        self.image = image
        self.label = label
        self.keep_prob = keep_prob
        self.prediction
        self.optimize
        self.accuracy

    @utils.define_scope
    def _pool_layer_1(self):
        """
        First pooling layer.
        """
        W_conv1 = utils.weight_variable([5, 5, 1, 32])
        b_conv1 = utils.bias_variable([32])
        x_image = tf.reshape(self.image, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(utils.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = utils.max_pool_2x2(h_conv1) # image size is 14 x 14 here
        return h_pool1

    @utils.define_scope
    def _pool_layer_2(self):
        """
        Second pooling layer.
        """
        W_conv2 = utils.weight_variable([5, 5, 32, 64])
        b_conv2 = utils.bias_variable([64])
        h_conv2 = tf.nn.relu(utils.conv2d(self._pool_layer_1, W_conv2) + b_conv2)
        h_pool2 = utils.max_pool_2x2(h_conv2) # image size is 7 x 7
        return h_pool2

    @utils.define_scope
    def _fully_connected_layer(self):
        """
        Fully connected layer with 1024 neurons to allow processing of entire image.
        """
        W_fc1 = utils.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = utils.bias_variable([1024])
        h_pool2_flat = tf.reshape(self._pool_layer_2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        return h_fc1

    @utils.define_scope
    def prediction(self):
        """
        Perform dropout (keep_prob is probability of neuron's output being kept)
        to prevent overfitting (also more useful when training very large networks).

        Then add readout layer.
        """
        # dropout layer
        h_fc1_drop = tf.nn.dropout(self._fully_connected_layer, self.keep_prob)

        # readout layer
        W_fc2 = utils.weight_variable([1024, 10])
        b_fc2 = utils.bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv

    @utils.define_scope
    def optimize(self):
        """
        Minimize cross entropy.
        """
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label, logits=self.prediction))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        return train_step


    @utils.define_scope
    def accuracy(self):
        """
        Accuracy calculation.
        """
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy


def main():
    """
    Load data, define placeholders, and train model.

    TODO:
    -Add saving points.
    """
    mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    image = tf.placeholder(tf.float32, shape=[None, 784])
    label = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    model = MNISTModel(image, label, keep_prob)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                writer = tf.summary.FileWriter('./graphs/mnist', sess.graph)
                train_accuracy = model.accuracy.eval(feed_dict={
                    image: batch[0], label: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            model.optimize.run(feed_dict={image: batch[0], label: batch[1], keep_prob: 0.5})
        print('test accuracy %g' % model.accuracy.eval(feed_dict={image: mnist.test.images,
            label: mnist.test.labels, keep_prob: 1.0}))

    writer.close()

if __name__ == '__main__':
    main()
