import math
import numpy as np
import tensorflow as tf

class SquareSumLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, input_shape, initializer = 'glorot_uniform'):
        super(SquareSumLayer, self).__init__()
        w_shape = input_shape
        w_shape.insert(0,1)

        if initializer.lower() == 'dft':
            with np.errstate(divide='ignore'):
                w = np.divide(1.0, np.square(np.arange(0,w_shape[1]//2)))
            w[w==np.inf] = 0
            wlong = np.reshape(np.array([w,w]), (1,w_shape[1]), order='F')
            self.ssweights = tf.Variable( wlong.astype(np.float32),
                    name="Square_Sum_Weights")
        else:
            if initializer.lower() == 'glorot_uniform':
                randfun = tf.keras.initializers.glorot_uniform()
            elif initializer.lower() == 'glorot_normal':
                randfun = tf.keras.initializers.glorot_normal()
            self.ssweights = tf.Variable(
                    randfun(w_shape),
                    name="Square_Sum_Weights")

        self.reduce_axis = range(1,len(w_shape))


    #==================================================================
    # Forward structure in the layer
    def call(self, in_data):
        n_data = tf.shape(in_data)[0]
        out = tf.math.square(in_data)
        out = tf.math.multiply(out, self.ssweights)
        out_data = tf.reduce_sum(out, axis=self.reduce_axis,
                keepdims=True)
        return(out_data)
