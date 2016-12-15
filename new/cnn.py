import tensorflow as tf
import numpy as np

import input

data = input.Process()

sess = tf.InteractiveSession()



class MCNN(object):
    """
    A CNN for mamography classification
    Uses 1 convolutional layer and 1 fully connected layer
    """
