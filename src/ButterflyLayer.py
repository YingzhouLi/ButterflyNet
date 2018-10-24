import math
import numpy as np
import tensorflow as tf

class ButterflyLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, out_siz, in_filter_siz = -1, out_filter_siz = -1,
            channel_siz = 12, nlvl = -1, prefixed = False):
        super(ButterflyLayer, self).__init__()
        self.out_siz        = out_siz
        self.in_filter_siz  = in_filter_siz
        self.out_filter_siz = out_filter_siz
        self.channel_siz    = channel_siz
        self.nlvl           = nlvl
        if prefixed:
            self.buildButterfly()
        else:
            self.buildRand()



    #==================================================================
    # Forward structure in the layer
    def call(self, in_data):
        InInterp = tf.nn.conv1d(in_data, self.InFilterVar,
                stride=self.in_filter_siz, padding='VALID')
        InInterp = tf.nn.relu(tf.nn.bias_add(InInterp, self.InBiasVar))

        tfVars = []
        for lvl in range(0,int(self.nlvl/2)):
            tmpVars = []
            if lvl > 0:
                for itk in range(0,2**(lvl+1)):
                    Var = tf.nn.conv1d(tfVars[lvl-1][math.floor(itk/2)],
                        self.FilterVars[lvl][itk],
                        stride=2, padding='VALID')
                    Var = tf.nn.relu(tf.nn.bias_add(Var,
                        self.BiasVars[lvl][itk]))
                    tmpVars.append(Var)
                tfVars.append(list(tmpVars))
            else:
                for itk in range(0,2**(lvl+1)):
                    Var = tf.nn.conv1d(InInterp,
                        self.FilterVars[lvl][itk],
                        stride=2, padding='VALID')
                    Var = tf.nn.relu(tf.nn.bias_add(Var,
                        self.BiasVars[lvl][itk]))
                    tmpVars.append(Var)
                tfVars.append(list(tmpVars))

        lvl = int(self.nlvl/2) - 1
        for itk in range(0,2**(int(self.nlvl/2))):
            tmpVars = np.reshape([], (np.size(in_data,0), 0,
                self.channel_siz))
            for itx in range(0,2**(int(self.nlvl/2))):
                tmpVar = tfVars[lvl][itk][:,itx,:]
                tmpVar = tf.matmul(tmpVar,self.MidDenseVars[itk][itx])
                tmpVar = tf.nn.relu( tf.nn.bias_add(
                    tmpVar, self.MidBiasVars[itk][itx] ) )
                tmpVar = tf.reshape(tmpVar,
                        (np.size(in_data,0),1,self.channel_siz))
                tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
            tfVars[lvl][itk] = tmpVars

        for lvl in range(int(self.nlvl/2),self.nlvl):
            tmpVars = []
            for itk in range(0,2**(lvl+1)):
                Var = tf.nn.conv1d(tfVars[lvl-1][math.floor(itk/2)],
                    self.FilterVars[lvl][itk],
                    stride=2, padding='VALID')
                Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl][itk]))
                tmpVars.append(Var)
            tfVars.append(list(tmpVars))

        # coef_filter of size filter_size*in_channels*out_channels
        lvl = self.nlvl-1
        OutInterp = np.reshape([],(np.size(in_data,0),1,0))
        for itk in range(0,2**(lvl+1)):
            Var = tf.nn.conv1d(tfVars[lvl][itk],
                self.OutFilterVars[itk],
                stride=1, padding='VALID')
            OutInterp = tf.concat([OutInterp, Var], axis=2)

        out_data = tf.reshape(OutInterp,shape=(np.size(in_data,0),
            self.out_siz,1))

        return(out_data)

    #==================================================================
    # Initialize variables in the layer
    def buildRand(self):
        self.InFilterVar = tf.Variable( tf.random_normal(
            [self.in_filter_siz, 1, self.channel_siz]),
            name="Filter_In" )
        self.InBiasVar = tf.Variable( tf.zeros([self.channel_siz]),
            name="Bias_In" )

        self.FilterVars = []
        self.BiasVars = []
        for lvl in range(0,self.nlvl):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**(lvl+1)):
                varLabel = "LVL_%02d_%04d" % (lvl, itk)
                filterVar = tf.Variable(
                        tf.random_normal([2,self.channel_siz,
                            self.channel_siz]),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))

        self.MidDenseVars = []
        self.MidBiasVars = []
        for itk in range(0,2**(int(self.nlvl/2))):
            tmpMidDenseVars = []
            tmpMidBiasVars = []
            for itx in range(0,2**(int(self.nlvl/2))):
                varLabel = "LVL_Mid_%04d_%04d" % (itk,itx)
                denseVar = tf.Variable(
                        tf.random_normal([self.channel_siz,
                            self.channel_siz]),
                        name="Dense_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpMidDenseVars.append(denseVar)
                tmpMidBiasVars.append(biasVar)
            self.MidDenseVars.append(list(tmpMidDenseVars))
            self.MidBiasVars.append(list(tmpMidBiasVars))

        self.OutFilterVars = []
        for itk in range(0,2**(self.nlvl)):
            varLabel = "Out_%04d" % (itk)
            filterVar = tf.Variable( tf.random_normal(
                [1, self.channel_siz, self.out_filter_siz]),
                name="Filter_"+varLabel )
            self.OutFilterVars.append(filterVar)

    #==================================================================
    # Initialize variables with coeffs in BF in the layer
    def buildButterfly(self):
        self.InFilterVar = tf.Variable( tf.random_normal(
            [self.in_filter_siz, 1, self.channel_siz]),
            name="Filter_In" )
        self.InBiasVar = tf.Variable( tf.zeros([self.channel_siz]),
            name="Bias_In" )

        self.FilterVars = []
        self.BiasVars = []
        for lvl in range(0,self.nlvl):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**(lvl+1)):
                varLabel = "LVL_%02d_%04d" % (lvl, itk)
                filterVar = tf.Variable(
                        tf.random_normal([2,self.channel_siz,
                            self.channel_siz]),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))

        self.MidDenseVars = []
        self.MidBiasVars = []
        for itk in range(0,2**(int(self.nlvl/2))):
            tmpMidDenseVars = []
            tmpMidBiasVars = []
            for itx in range(0,2**(int(self.nlvl/2))):
                varLabel = "LVL_Mid_%04d_%04d" % (itk,itx)
                denseVar = tf.Variable(
                        tf.random_normal([self.channel_siz,
                            self.channel_siz]),
                        name="Dense_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpMidDenseVars.append(denseVar)
                tmpMidBiasVars.append(biasVar)
            self.MidDenseVars.append(list(tmpMidDenseVars))
            self.MidBiasVars.append(list(tmpMidBiasVars))

        self.OutFilterVars = []
        for itk in range(0,2**(self.nlvl)):
            varLabel = "Out_%04d" % (itk)
            filterVar = tf.Variable( tf.random_normal(
                [1, self.channel_siz, self.out_filter_siz]),
                name="Filter_"+varLabel )
            self.OutFilterVars.append(filterVar)
