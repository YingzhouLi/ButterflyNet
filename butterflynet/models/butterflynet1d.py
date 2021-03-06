import sys
import tensorflow as tf

from .. import layers

class ButterflyNet1D(tf.keras.Model):
    def __init__(self, in_siz, out_siz, inouttype,
            channel_siz, nlvl = -1, nlvlx = -1, nlvlk = -1,
            initializer = 'glorot_uniform',
            in_range = [], out_range = [],
            task_layer_type = "", task_units = 0):
        super(ButterflyNet1D, self).__init__()
        self.blayer1d = layers.ButterflyLayer1D(in_siz, out_siz,
                inouttype, channel_siz, nlvl, nlvlx, nlvlk,
                initializer, in_range, out_range)
        self.task_layer_type = task_layer_type
        if self.task_layer_type.lower() == 'squaresum':
            self.squaresumlayer = layers.SquareSumLayer([out_siz],
                    initializer)
        elif self.task_layer_type.lower() == 'singlefullyconnect':
            self.singlefullyconnectlayer = \
                    layers.SingleFullyConnectLayer([out_siz],
                            task_units)

    def call(self, in_data):
        out = self.blayer1d(in_data)
        if self.task_layer_type.lower() == 'squaresum':
            out_data = self.squaresumlayer(out)
        elif self.task_layer_type.lower() == 'singlefullyconnect':
            out_data = self.singlefullyconnectlayer(out)
        else:
            out_data = out
        return out_data

    def summary(self, output_stream = sys.stdout):
        print("================== ButterflyNet1D Summary ==================")
        print("num of layers:                %30d" % (self.blayer1d.L))
        print("num of layers before switch:  %30d" % (self.blayer1d.Lx))
        print("    branching:                %30d" % (self.blayer1d.Lx1))
        print("    fixed:                    %30d" % (self.blayer1d.Lx2))
        print("num of layers after switch:   %30d" % (self.blayer1d.Lk))
        print("    fixed:                    %30d" % (self.blayer1d.Lk3))
        print("    branching:                %30d" % (self.blayer1d.Lk4))
        print("------------------------------------------------------------")
        print("Parameter Count")
        n_paras = tf.size(self.blayer1d.XFilterVar).numpy() \
                + tf.size(self.blayer1d.XBiasVar).numpy()
        tot_n_paras = n_paras
        print("    Interpolation    0:       %30d" % (n_paras))

        for lvl in range(1,self.blayer1d.Lx+1):
            n_paras = 0
            for it in range(len(self.blayer1d.FilterVars[lvl])):
                n_paras = n_paras \
                    + tf.size(self.blayer1d.FilterVars[lvl][it]).numpy() \
                    + tf.size(self.blayer1d.BiasVars[lvl][it]).numpy()
            tot_n_paras = tot_n_paras + n_paras
            print("    Recursion      %3d:       %30d" % (lvl,n_paras))

        n_paras = 0
        for itk in range(len(self.blayer1d.MidDenseVars)):
            for itx in range(len(self.blayer1d.MidDenseVars[itk])):
                n_paras = n_paras \
                    + tf.size(self.blayer1d.MidDenseVars[itk][itx]).numpy()\
                    + tf.size(self.blayer1d.MidBiasVars[itk][itx]).numpy()
        tot_n_paras = tot_n_paras + n_paras
        print("    Switch            :       %30d" % (n_paras))

        for lvl in range(self.blayer1d.Lx+1,self.blayer1d.L+1):
            n_paras = 0
            for it in range(len(self.blayer1d.FilterVars[lvl])):
                n_paras = n_paras \
                    + tf.size(self.blayer1d.FilterVars[lvl][it]).numpy() \
                    + tf.size(self.blayer1d.BiasVars[lvl][it]).numpy()
            tot_n_paras = tot_n_paras + n_paras
            print("    Recursion      %3d:       %30d" % (lvl,n_paras))

        n_paras = tf.size(self.blayer1d.KFilterVar).numpy()
        tot_n_paras = tot_n_paras + n_paras
        print("    Interpolation  %3d:       %30d" \
                % (self.blayer1d.L+1,n_paras))

        if self.task_layer_type.lower() == 'squaresum':
            n_paras = tf.size(self.squaresumlayer.ssweights).numpy()
            tot_n_paras = tot_n_paras + n_paras
            print("    Task              :       %30d" \
                    % (n_paras))
        elif self.task_layer_type.lower() == 'singlefullyconnect':
            n_paras = tf.size(self.singlefullyconnectlayer.Mat1).numpy() \
                + tf.size(self.singlefullyconnectlayer.Bias1).numpy() \
                + tf.size(self.singlefullyconnectlayer.Mat2).numpy()
            tot_n_paras = tot_n_paras + n_paras
            print("    Task              :       %30d" \
                    % (n_paras))
        print("Total number of parameters    %30d" % tot_n_paras)
        print("============================================================")
        print("")
