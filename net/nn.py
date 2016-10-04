'''

The ideia here is to implement a neural network with similar structute to, the already known, AlexNet from the ImageNet Competition 2012.
The NN has 8 main layers, being the first 5 convolutional and the 3 left fully connected layers.



    author: Bruno Smarsaro Bazelato

'''

import tensorflow as tf
sess = tf.InteractiveSession()

# TODO: to prepare input data accordingly