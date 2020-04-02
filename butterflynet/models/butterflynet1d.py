import tensorflow as tf

from .. import layers

class ButterflyNet1D(tf.keras.Model):
    def __init__(self, in_siz, out_siz, inouttype,
            channel_siz, nlvl = -1, nlvlx = -1, nlvlk = -1,
            prefixed = False, in_range = [], out_range = []):
        super(ButterflyNet1D, self).__init__()
        self.blayer1d = layers.ButterflyLayer1D(in_siz, out_siz,
                inouttype, channel_siz, nlvl, nlvlx, nlvlk,
                prefixed, in_range, out_range)

    def call(self, in_data):
        return self.blayer1d(in_data)
