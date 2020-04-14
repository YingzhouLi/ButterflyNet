import math
import numpy as np
import tensorflow as tf

class SingleFullyConnectLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, input_shape, units, initializer = 'glorot_uniform'):
        super(SingleFullyConnectLayer, self).__init__()

        if initializer.lower() == 'glorot_uniform':
            randfun = tf.keras.initializers.glorot_uniform()
        elif initializer.lower() == 'glorot_normal':
            randfun = tf.keras.initializers.glorot_normal()
        else:
            randfun = tf.keras.initializers.glorot_uniform()

        self.in_siz = np.prod(input_shape)
        self.units = units
        self.Mat1 = tf.Variable(randfun([self.in_siz, units]),
                name="SFC Mat1")
        self.Bias1 = tf.Variable(randfun([units]),
                name="SFC Bias1")
        self.Mat2 = tf.Variable(randfun([units,1]),
                name="SFC Mat2")

    #==================================================================
    # Forward structure in the layer
    def call(self, in_data):
        n_data = tf.shape(in_data)[0]
        out = tf.reshape(in_data,[n_data, self.in_siz])
        out = tf.matmul(out, self.Mat1)
        out = tf.nn.relu( tf.nn.bias_add(out, self.Bias1 ) )
        out_data = tf.matmul(out, self.Mat2)
        return(out_data)
