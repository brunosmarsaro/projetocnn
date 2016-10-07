"""
The idea here is to implement a neural network with similar structute to, the already known, AlexNet from the ImageNet Competition 2012.
The NN has 8 main layers, being the first 5 convolutional and the 3 left fully connected layers.



    author: Bruno Smarsaro Bazelato

"""

import tensorflow as tf

import gen_input

class Deepnn:
    def __init__(self, data_dir=None, x_shape_size=1024, testing_percetage=10, validation_percentage=10):
        self.t = gen_input.process()
        if data_dir is not None: self.t.set_path(data_dir)
        self.t.gen_labels()
        y_shape_size = len(self.t.labels) -1

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, x_shape_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, y_shape_size])

        # Parameters
        self.testing_percetage = testing_percetage
        self.validation_percentage = validation_percentage

    # Variables initialisation
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Convolution and Pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    # Set the data array for training, testing and validation
    def set_data(self):
        dic = self.t.create_image_lists(self.testing_percetage, self.validation_percentage)
        print(dic)