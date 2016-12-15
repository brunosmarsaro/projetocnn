"""

    author: Bruno Smarsaro Bazelato
"""

import tensorflow as tf
import numpy as np

import gen_input


class DeepNN:
    """
        Attributes:
            data_dir: directory to the database
            x_shape_size: size of the input (number of pixels)
            y_shape_size: number of output classes
            testing_percentage: percentage of testing data
            validation_percentage: percentage of validation data
            x: placeholder
            y_ : placeholder
            t: instance of gen_input class to generate set data.
            inputs: input set
            index:

        Methods:
            weight_variable and bias_variable: initiate variables
            conv2d: convolution
            max_pool_2x2: pooling
            set_data: reload dataset
            next_batch: select new set of data to be given as input during a step

    """
    def __init__(self, data_dir=None, x_shape_size=1024*1024, testing_percentage=10, validation_percentage=0):
        self.t = gen_input.Process()
        if data_dir is not None: self.t.set_path(data_dir)
        self.t.gen_labels()
        y_shape_size = len(self.t.labels) -1

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, x_shape_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, y_shape_size])

        # Parameters
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage

        # Setting Inputs
        self.inputs = self.t.run(self.testing_percentage, self.validation_percentage)
        self.index = 0

    # Variables initialisation
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Convolution and Pooling
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    # Set the data array for training, testing and validation
    def set_data(self):
        self.inputs = self.t.run()

    def next_batch(self, data, size):
        start = self.index
        self.index += size
        if self.index > data[0].shape[0]:
            perm = np.arange(data[0].shape[0])
            np.random.shuffle(perm)
            data[0] = data[0][perm]
            data[1] = data[1][perm]
            start = 0
            self.index = size
            #print(data[0].shape[0])
            #input()
            #assert size <= data[0].shape[0]
        end = self.index
        return data[0][start:end], data[1][start:end]