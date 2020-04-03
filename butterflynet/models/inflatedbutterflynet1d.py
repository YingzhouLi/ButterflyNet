import sys
import tensorflow as tf

from .. import layers

class InflatedButterflyNet1D(tf.keras.Model):
    def __init__(self, in_siz, out_siz, inouttype,
            channel_siz, nlvl = -1, nlvlx = -1, nlvlk = -1,
            prefixed = False, in_range = [], out_range = []):
        super(InflatedButterflyNet1D, self).__init__()
        self.iblayer1d = layers.InflatedButterflyLayer1D(in_siz, out_siz,
                inouttype, channel_siz, nlvl, nlvlx, nlvlk,
                prefixed, in_range, out_range)

    def call(self, in_data):
        return self.iblayer1d(in_data)

    def summary(self, output_stream = sys.stdout):
        print("======== ButterflyNet1D Summary ===========")
        print("num of layers:                %6d" % (self.iblayer1d.L))
        print("num of layers before switch:  %6d" % (self.iblayer1d.Lx))
        print("    branching:                %6d" % (self.iblayer1d.Lx1))
        print("    fixed:                    %6d" % (self.iblayer1d.Lx2))
        print("num of layers after switch:   %6d" % (self.iblayer1d.Lk))
        print("    fixed:                    %6d" % (self.iblayer1d.Lk3))
        print("    branching:                %6d" % (self.iblayer1d.Lk4))
        print("")

        print("Number of parameters")
        n_paras = tf.size(self.iblayer1d.XFilterVar).numpy() \
                + tf.size(self.iblayer1d.XBiasVar).numpy()
        tot_n_paras = n_paras
        print("    Interpolation  0:     %10d" % (n_paras))

        for lvl in range(1,self.iblayer1d.Lx+1):
            n_paras = tf.size(self.iblayer1d.FilterVars[lvl]).numpy() \
                + tf.size(self.iblayer1d.BiasVars[lvl]).numpy()
            tot_n_paras = tot_n_paras + n_paras
            print("    Recursion     %2d:     %10d" % (lvl,n_paras))

        n_paras = 0
        for itk in range(len(self.iblayer1d.MidDenseVars)):
            for itx in range(len(self.iblayer1d.MidDenseVars[itk])):
                n_paras = n_paras \
                    +tf.size(self.iblayer1d.MidDenseVars[itk][itx]).numpy()\
                    +tf.size(self.iblayer1d.MidBiasVars[itk][itx]).numpy()
        tot_n_paras = tot_n_paras + n_paras
        print("    Switch          :     %10d" % (n_paras))

        for lvl in range(self.iblayer1d.Lx+1,self.iblayer1d.L+1):
            n_paras = tf.size(self.iblayer1d.FilterVars[lvl]).numpy() \
               + tf.size(self.iblayer1d.BiasVars[lvl]).numpy()
            tot_n_paras = tot_n_paras + n_paras
            print("    Recursion     %2d:     %10d" % (lvl,n_paras))

        n_paras = tf.size(self.iblayer1d.KFilterVar).numpy()
        tot_n_paras = tot_n_paras + n_paras
        print("    Interpolation %2d:     %10d" \
                % (self.iblayer1d.L+1,n_paras))
        print("total num of paras      %12d" % tot_n_paras)
        print("")
