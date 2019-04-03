import math
import numpy as np
import tensorflow as tf

class CNNLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, in_siz, out_siz,
            in_filter_siz = -1, out_filter_siz = -1,
            channel_siz = 8, nlvl = -1, prefixed = True):
        super(CNNLayer, self).__init__()
        self.in_siz         = in_siz
        self.out_siz        = out_siz
        self.in_filter_siz  = in_filter_siz
        self.out_filter_siz = out_filter_siz
        self.channel_siz    = channel_siz
        self.nlvl           = nlvl
        if prefixed:
            self.buildButterflyCNN()
        else:
            self.buildRand()


    #==================================================================
    # Forward structure in the layer
    def call(self, in_data):
        InInterp = tf.nn.conv1d(in_data, self.InFilterVar,
                stride=self.in_filter_siz, padding='VALID')
        InInterp = tf.nn.relu(tf.nn.bias_add(InInterp, self.InBiasVar))

        tfVars = []
        tfVars.append(InInterp)

        for lvl in range(1,self.nlvl//2+1):
            Var = tf.nn.conv1d(tfVars[lvl-1],
                    self.FilterVars[lvl],
                    stride=2, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl]))
            tfVars.append(Var)

        # Middle level
        lvl = self.nlvl//2
        tmpVarsk = np.reshape([], (np.size(in_data,0), 2**lvl, 0))
        for itx in range(0,2**(self.nlvl-lvl)):
            tmpVars = np.reshape([], (np.size(in_data,0), 0,
                self.channel_siz))
            for itk in range(0,2**lvl):
                tmpVar = tfVars[lvl][:,itx,
                        itk*self.channel_siz : (itk+1)*self.channel_siz ]
                tmpVar = tf.matmul(tmpVar,self.MidDenseVars[itk][itx])
                tmpVar = tf.nn.relu( tf.nn.bias_add(
                    tmpVar, self.MidBiasVars[itk][itx] ) )
                tmpVar = tf.reshape(tmpVar,
                        (np.size(in_data,0),1,self.channel_siz))
                tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
            tmpVarsk = tf.concat([tmpVarsk, tmpVars], axis=2)
        tfVars[lvl] = tmpVarsk

        for lvl in range(self.nlvl//2+1,self.nlvl+1):
            tmptfVars = np.reshape([], (np.size(in_data,0), 2**lvl, 0))
            for itx in range(0,2**(self.nlvl-lvl)):
                tmpVars = np.reshape([], (np.size(in_data,0), 0,
                    self.channel_siz))
                for itk in range(0,2**(lvl-1)):
                    tmpVar = tfVars[lvl-1][:, itk,
                        2*itx*self.channel_siz:(2*itx+2)*self.channel_siz ]
                    tmpVar = tf.reshape(tmpVar,
                        (np.size(in_data,0),2,self.channel_siz))
                    tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
                tmptfVars = tf.concat([tmptfVars, tmpVars], axis=2)
            Var = tf.nn.conv1d(tmptfVars,
                    self.FilterVars[lvl],
                    stride=2, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl]))
            tmpVarsk = np.reshape([], (np.size(in_data,0),
                2**lvl,0))
            for itx in range(0,2**(self.nlvl-lvl)):
                tmpVars = np.reshape([], (np.size(in_data,0), 0,
                    self.channel_siz))
                for itk in range(0,2**(lvl-1)):
                    tmpVar = Var[:, itk, (2*itx)*self.channel_siz 
                            : (2*itx+2)*self.channel_siz ]
                    tmpVar = tf.reshape(tmpVar,
                            (np.size(in_data,0),2,self.channel_siz))
                    tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
                tmpVarsk = tf.concat([tmpVarsk, tmpVars], axis=2)
            tfVars.append(tmpVarsk)

        # coef_filter of size filter_size*in_channels*out_channels
        OutInterp = tf.nn.conv1d(tfVars[self.nlvl],
            self.OutFilterVar, stride=1, padding='VALID')

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
        self.FilterVars.append([])
        self.BiasVars.append([])
        for lvl in range(1,self.nlvl//2+1):
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable(
                    tf.random_normal([2,2**(lvl-1)*self.channel_siz,
                        2**lvl*self.channel_siz]),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**lvl*self.channel_siz]),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        for lvl in range(self.nlvl//2+1,self.nlvl+1):
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable(
                    tf.random_normal([2,
                        2**(self.nlvl-lvl)*self.channel_siz,
                        2**(self.nlvl-lvl+1)*self.channel_siz]),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([
                2**(self.nlvl-lvl+1)*self.channel_siz]),
                name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        self.MidDenseVars = []
        self.MidBiasVars = []
        lvl = self.nlvl//2
        for itk in range(0,2**lvl):
            tmpMidDenseVars = []
            tmpMidBiasVars = []
            for itx in range(0,2**(self.nlvl-lvl)):
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


        self.OutFilterVar = tf.Variable( tf.random_normal(
                [1, self.channel_siz, self.out_filter_siz]),
                name="Filter_Out" )

    #==================================================================
    # Initialize variables with coeffs in BF in the layer
    def buildButterflyCNN(self):
        NG = int(self.channel_siz/4)

        #----------------
        # Setup initial interpolation weights
        mat = np.load('tftmp/Filter_In.npy')
        self.InFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_In" )
        mat = np.load('tftmp/Bias_In.npy')
        self.InBiasVar = tf.Variable( mat.astype(np.float32),
                name="Bias_In" )

        #----------------
        # Setup right factor interpolation weights
        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append([])
        self.BiasVars.append([])
        for lvl in range(1,self.nlvl//2+1):
            bigmatf = np.zeros((2, 2**(lvl-1)*self.channel_siz,
                    2**lvl*self.channel_siz))
            bigmatb = np.zeros((2**lvl*self.channel_siz))
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d" % (lvl, itk)
                offset1 = itk//2*self.channel_siz
                offset2 = itk*self.channel_siz
                siz = self.channel_siz
                mat = np.load('tftmp/Filter_'+varLabel+'.npy')
                bigmatf[:, offset1:offset1+siz,
                        offset2:offset2+siz] = mat
                mat = np.load('tftmp/Bias_'+varLabel+'.npy')
                bigmatb[offset2:offset2+siz] = mat
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable( bigmatb.astype(np.float32),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        #----------------
        # Setup weights for middle layer
        self.MidDenseVars = []
        self.MidBiasVars = []
        lvl = self.nlvl//2
        for itk in range(0,2**lvl):
            tmpMidDenseVars = []
            tmpMidBiasVars = []
            for itx in range(0,2**(self.nlvl-lvl)):
                varLabel = "LVL_Mid_%04d_%04d" % (itk,itx)
                mat = np.load('tftmp/Dense_'+varLabel+'.npy')
                denseVar = tf.Variable( mat.astype(np.float32),
                        name="Dense_"+varLabel )
                mat = np.load('tftmp/Bias_'+varLabel+'.npy')
                biasVar = tf.Variable( mat.astype(np.float32),
                        name="Bias_"+varLabel )
                tmpMidDenseVars.append(denseVar)
                tmpMidBiasVars.append(biasVar)
            self.MidDenseVars.append(list(tmpMidDenseVars))
            self.MidBiasVars.append(list(tmpMidBiasVars))

        #----------------
        # Setup left factor interpolation weights
        for lvl in range(int(self.nlvl/2)+1,self.nlvl+1):
            bigmatf = np.zeros((2, 2**(self.nlvl-lvl)*self.channel_siz,
                    2**(self.nlvl-lvl+1)*self.channel_siz))
            bigmatb = np.zeros((2**(self.nlvl-lvl+1)*self.channel_siz))
            for itx in range(0,2**(self.nlvl-lvl+1)):
                varLabel = "LVL_%02d_%04d" % (lvl, itx)
                offset1 = itx//2*self.channel_siz
                offset2 = itx*self.channel_siz
                siz = self.channel_siz
                mat = np.load('tftmp/Filter_'+varLabel+'.npy')
                bigmatf[:, offset1:offset1+siz,
                        offset2:offset2+siz] = mat
                mat = np.load('tftmp/Bias_'+varLabel+'.npy')
                bigmatb[offset2:offset2+siz] = mat
                filterVar = tf.Variable( bigmatf.astype(np.float32),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable( bigmatb.astype(np.float32),
                        name="Bias_"+varLabel )
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable( bigmatb.astype(np.float32),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        #----------------
        # Setup final interpolation weights
        mat = np.load('tftmp/Filter_Out.npy')
        self.OutFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_Out" )
