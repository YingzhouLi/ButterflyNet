import math
import numpy as np
import tensorflow as tf

from LagrangeMat import LagrangeMat

class ButterflyLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, in_siz, out_siz,
            in_filter_siz = -1, out_filter_siz = -1,
            channel_siz = 8, nlvl = -1, prefixed = False,
            in_range = [], out_range = []):
        super(ButterflyLayer, self).__init__()
        self.in_siz         = in_siz
        self.out_siz        = out_siz
        #TODO: set the default values based on in_siz and out_siz
        self.in_filter_siz  = in_filter_siz
        self.out_filter_siz = out_filter_siz
        self.channel_siz    = channel_siz
        self.nlvl           = nlvl
        self.in_range       = in_range
        self.out_range      = out_range
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
        tmpVars = []
        tmpVars.append(InInterp)
        tfVars.append(list(tmpVars))

        for lvl in range(1,int(self.nlvl/2)+1):
            tmpVars = []
            for itk in range(0,2**lvl):
                Var = tf.nn.conv1d(tfVars[lvl-1][math.floor(itk/2)],
                    self.FilterVars[lvl][itk],
                    stride=2, padding='VALID')
                Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl][itk]))
                tmpVars.append(Var)
            tfVars.append(list(tmpVars))

        # Middle level
        lvl = int(self.nlvl/2)
        for itk in range(0,2**lvl):
            tmpVars = np.reshape([], (np.size(in_data,0), 0,
                self.channel_siz))
            for itx in range(0,2**(self.nlvl-lvl)):
                tmpVar = tfVars[lvl][itk][:,itx,:]
                tmpVar = tf.matmul(tmpVar,self.MidDenseVars[itk][itx])
                tmpVar = tf.nn.relu( tf.nn.bias_add(
                    tmpVar, self.MidBiasVars[itk][itx] ) )
                tmpVar = tf.reshape(tmpVar,
                        (np.size(in_data,0),1,self.channel_siz))
                tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
            tfVars[lvl][itk] = tmpVars

        # Reorganize before conv1d
        lvl = int(self.nlvl/2)
        tmptfVars = []
        for itx in range(0,2**(self.nlvl-lvl-1)):
            tmpVars = np.reshape([], (np.size(in_data,0), 0,
                self.channel_siz))
            for itk in range(0,2**lvl):
                tmpVar = tf.reshape(tfVars[lvl][itk][:,2*itx,:],
                    (np.size(in_data,0),1,self.channel_siz))
                tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
                tmpVar = tf.reshape(tfVars[lvl][itk][:,2*itx+1,:],
                    (np.size(in_data,0),1,self.channel_siz))
                tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
            tmptfVars.append(tmpVars)
        tfVars[lvl] = tmptfVars

        for lvl in range(int(self.nlvl/2)+1,self.nlvl):
            tmpVars = []
            for itx in range(0,2**(self.nlvl-lvl)):
                if itx % 2 == 0:
                    Var01 = tf.nn.conv1d(tfVars[lvl-1][itx],
                        self.FilterVars[lvl][2*itx],
                        stride=2, padding='VALID')
                    Var01 = tf.nn.relu(tf.nn.bias_add(Var01,
                        self.BiasVars[lvl][2*itx]))
                    Var02 = tf.nn.conv1d(tfVars[lvl-1][itx],
                        self.FilterVars[lvl][2*itx+1],
                        stride=2, padding='VALID')
                    Var02 = tf.nn.relu(tf.nn.bias_add(Var02,
                        self.BiasVars[lvl][2*itx+1]))
                else:
                    Var11 = tf.nn.conv1d(tfVars[lvl-1][itx],
                        self.FilterVars[lvl][2*itx],
                        stride=2, padding='VALID')
                    Var11 = tf.nn.relu(tf.nn.bias_add(Var11,
                        self.BiasVars[lvl][2*itx]))
                    Var12 = tf.nn.conv1d(tfVars[lvl-1][itx],
                        self.FilterVars[lvl][2*itx+1],
                        stride=2, padding='VALID')
                    Var12 = tf.nn.relu(tf.nn.bias_add(Var12,
                        self.BiasVars[lvl][2*itx+1]))

                    Vars = np.reshape([], (np.size(in_data,0), 0,
                        self.channel_siz))
                    for itk in range(0,2**(lvl-1)):
                        Var = tf.reshape(Var01[:,itk,:],
                            (np.size(in_data,0),1,self.channel_siz))
                        Vars = tf.concat([Vars, Var], axis=1)
                        Var = tf.reshape(Var11[:,itk,:],
                            (np.size(in_data,0),1,self.channel_siz))
                        Vars = tf.concat([Vars, Var], axis=1)
                        Var = tf.reshape(Var02[:,itk,:],
                            (np.size(in_data,0),1,self.channel_siz))
                        Vars = tf.concat([Vars, Var], axis=1)
                        Var = tf.reshape(Var12[:,itk,:],
                            (np.size(in_data,0),1,self.channel_siz))
                        Vars = tf.concat([Vars, Var], axis=1)
                    tmpVars.append(Vars)
            tfVars.append(list(tmpVars))

        lvl = self.nlvl
        tmpVars = []
        Var1 = tf.nn.conv1d(tfVars[lvl-1][0],
            self.FilterVars[lvl][0],
            stride=2, padding='VALID')
        Var1 = tf.nn.relu(tf.nn.bias_add(Var1,
            self.BiasVars[lvl][0]))
        Var2 = tf.nn.conv1d(tfVars[lvl-1][0],
            self.FilterVars[lvl][1],
            stride=2, padding='VALID')
        Var2 = tf.nn.relu(tf.nn.bias_add(Var2,
            self.BiasVars[lvl][1]))

        Vars = np.reshape([], (np.size(in_data,0), 0,
            self.channel_siz))
        for itk in range(0,2**(lvl-1)):
            Var = tf.reshape(Var1[:,itk,:],
                (np.size(in_data,0),1,self.channel_siz))
            Vars = tf.concat([Vars, Var], axis=1)
            Var = tf.reshape(Var2[:,itk,:],
                (np.size(in_data,0),1,self.channel_siz))
            Vars = tf.concat([Vars, Var], axis=1)
        tmpVars.append(Vars)
        tfVars.append(list(tmpVars))

        # coef_filter of size filter_size*in_channels*out_channels
        OutInterp = tf.nn.conv1d(tfVars[self.nlvl][0],
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
        self.FilterVars.append(list([]))
        self.BiasVars.append(list([]))
        for lvl in range(1,int(self.nlvl/2)+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
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

        for lvl in range(int(self.nlvl/2)+1,self.nlvl+1):
            for itx in range(0,2**(self.nlvl-lvl)):
                varLabel = "LVL_%02d_%04d" % (lvl, itx)
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
        lvl = int(self.nlvl/2)
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
    def buildButterfly(self):
        NG = int(self.channel_siz/4)
        ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) +
                1)/2
        xNodes = np.arange(0,1,1.0/self.in_filter_siz)
        LMat = LagrangeMat(ChebNodes,xNodes)

        #----------------
        # Setup initial interpolation weights
        mat = np.empty((self.in_filter_siz,1,self.channel_siz))
        kcen = np.mean(self.out_range)
        xlen = (self.in_range[1] - self.in_range[0])/2**self.nlvl
        for it in range(0,NG):
            KVal = np.exp(-2*math.pi*1j*kcen*(xNodes-ChebNodes[it])*xlen)
            LVec = np.squeeze(LMat[:,it])
            mat[:,0,4*it]   =  np.multiply(KVal.real,LVec)
            mat[:,0,4*it+1] =  np.multiply(KVal.imag,LVec)
            mat[:,0,4*it+2] = -np.multiply(KVal.real,LVec)
            mat[:,0,4*it+3] = -np.multiply(KVal.imag,LVec)

        self.InFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_In" )
        self.InBiasVar = tf.Variable( tf.zeros([self.channel_siz]),
                name="Bias_In" )

        #----------------
        # Setup right factor interpolation weights
        ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) +
                1)/2
        x1Nodes = ChebNodes/2
        x2Nodes = ChebNodes/2 + 1/2
        LMat1 = LagrangeMat(ChebNodes,x1Nodes)
        LMat2 = LagrangeMat(ChebNodes,x2Nodes)

        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append(list([]))
        self.BiasVars.append(list([]))
        for lvl in range(1,int(self.nlvl/2)+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d" % (lvl, itk)

                mat = np.empty((2, self.channel_siz, self.channel_siz))
                kcen = (self.out_range[1] \
                        - self.out_range[0])/2**lvl*(itk+0.5) \
                        + self.out_range[0]
                xlen = (self.in_range[1] - \
                        self.in_range[0])/2**(self.nlvl-lvl)
                for it in range(0,NG):
                    KVal = np.exp( -2*math.pi*1j * kcen *
                            (x1Nodes-ChebNodes[it]) * xlen)
                    LVec = np.squeeze(LMat1[:,it])
                    mat[0,range(0,4*NG,4),4*it  ] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(1,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(2,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,range(3,4*NG,4),4*it  ] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(0,4*NG,4),4*it+1] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(1,4*NG,4),4*it+1] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(2,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(3,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,:,(4*it+2,4*it+3)] = - mat[0,:,(4*it,4*it+1)]

                    KVal = np.exp( -2*math.pi*1j * kcen *
                            (x2Nodes-ChebNodes[it]) * xlen)
                    LVec = np.squeeze(LMat2[:,it])
                    mat[1,range(0,4*NG,4),4*it  ] = \
                            np.multiply(KVal.real,LVec)
                    mat[1,range(1,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[1,range(2,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.real,LVec)
                    mat[1,range(3,4*NG,4),4*it  ] = \
                            np.multiply(KVal.imag,LVec)
                    mat[1,range(0,4*NG,4),4*it+1] = \
                            np.multiply(KVal.imag,LVec)
                    mat[1,range(1,4*NG,4),4*it+1] = \
                            np.multiply(KVal.real,LVec)
                    mat[1,range(2,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[1,range(3,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.real,LVec)
                    mat[1,:,(4*it+2,4*it+3)] = - mat[1,:,(4*it,4*it+1)]

                filterVar = tf.Variable( mat.astype(np.float32),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))

        #----------------
        # Setup weights for middle layer
        self.MidDenseVars = []
        self.MidBiasVars = []
        lvl = int(self.nlvl/2)
        for itk in range(0,2**lvl):
            tmpMidDenseVars = []
            tmpMidBiasVars = []
            for itx in range(0,2**(self.nlvl-lvl)):
                varLabel = "LVL_Mid_%04d_%04d" % (itk,itx)

                mat = np.empty((self.channel_siz, self.channel_siz))
                klen = (self.out_range[1] - self.out_range[0])/2**lvl
                koff = klen*itk + self.out_range[0]
                kNodes = ChebNodes*klen + koff
                xlen = (self.in_range[1] - \
                        self.in_range[0])/2**(self.nlvl-lvl)
                xoff = xlen*itx + self.in_range[0]
                xNodes = ChebNodes*xlen + xoff

                for iti in range(0,NG):
                    for itj in range(0,NG):
                        KVal = np.exp( - 2*math.pi*1j
                                *kNodes[itj]*xNodes[iti])
                        mat[4*iti  ,4*itj  ] =   KVal.real
                        mat[4*iti+1,4*itj  ] = - KVal.imag
                        mat[4*iti+2,4*itj  ] = - KVal.real
                        mat[4*iti+3,4*itj  ] =   KVal.imag
                        mat[4*iti  ,4*itj+1] =   KVal.imag
                        mat[4*iti+1,4*itj+1] =   KVal.real
                        mat[4*iti+2,4*itj+1] = - KVal.imag
                        mat[4*iti+3,4*itj+1] = - KVal.real
                        mat[:,(4*itj+2,4*itj+3)] = - mat[:,(4*itj,4*itj+1)]

                denseVar = tf.Variable( mat.astype(np.float32),
                        name="Dense_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpMidDenseVars.append(denseVar)
                tmpMidBiasVars.append(biasVar)
            self.MidDenseVars.append(list(tmpMidDenseVars))
            self.MidBiasVars.append(list(tmpMidBiasVars))

        #----------------
        # Setup left factor interpolation weights
        ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) +
                1)/2
        k1Nodes = ChebNodes/2
        k2Nodes = ChebNodes/2 + 1/2
        LMat1 = LagrangeMat(ChebNodes,k1Nodes)
        LMat2 = LagrangeMat(ChebNodes,k2Nodes)

        for lvl in range(int(self.nlvl/2)+1,self.nlvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itx in range(0,2**(self.nlvl-lvl+1)):
                varLabel = "LVL_%02d_%04d" % (lvl, itx)
                mat = np.empty((2, self.channel_siz, self.channel_siz))
                if itx % 2 == 0:
                    LMat = LMat1
                    kNodes = k1Nodes
                else:
                    LMat = LMat2
                    kNodes = k2Nodes
                klen = (self.out_range[1] - \
                        self.out_range[0])/2**lvl

                for it in range(0,NG):
                    trueitx = int(itx/2)*2
                    xcen = (self.in_range[1] \
                        - self.in_range[0]) \
                        / 2**(self.nlvl-lvl)*(trueitx+0.5) \
                        + self.in_range[0]
                    KVal = np.exp( -2*math.pi*1j * xcen *
                            (kNodes[it]-ChebNodes) * klen)
                    LVec = np.squeeze(LMat[it,:])
                    mat[0,range(0,4*NG,4),4*it  ] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(1,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(2,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,range(3,4*NG,4),4*it  ] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(0,4*NG,4),4*it+1] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(1,4*NG,4),4*it+1] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(2,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(3,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,:,(4*it+2,4*it+3)] = - mat[0,:,(4*it,4*it+1)]

                    trueitx = int(itx/2)*2+1
                    xcen = (self.in_range[1] \
                        - self.in_range[0]) \
                        / 2**(self.nlvl-lvl)*(trueitx+0.5) \
                        + self.in_range[0]
                    KVal = np.exp( -2*math.pi*1j * xcen *
                            (kNodes[it]-ChebNodes) * klen)
                    mat[1,range(0,4*NG,4),4*it  ] = \
                            np.multiply(KVal.real,LVec)
                    mat[1,range(1,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[1,range(2,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.real,LVec)
                    mat[1,range(3,4*NG,4),4*it  ] = \
                            np.multiply(KVal.imag,LVec)
                    mat[1,range(0,4*NG,4),4*it+1] = \
                            np.multiply(KVal.imag,LVec)
                    mat[1,range(1,4*NG,4),4*it+1] = \
                            np.multiply(KVal.real,LVec)
                    mat[1,range(2,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[1,range(3,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.real,LVec)
                    mat[1,:,(4*it+2,4*it+3)] = - mat[1,:,(4*it,4*it+1)]

                filterVar = tf.Variable( mat.astype(np.float32),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))

        kNodes = np.arange(0,1,2.0/self.out_filter_siz)
        LMat = LagrangeMat(ChebNodes,kNodes)

        #----------------
        # Setup final interpolation weights
        mat = np.empty((1,self.channel_siz,self.out_filter_siz))
        xcen = np.mean(self.in_range)
        klen = (self.out_range[1] - self.out_range[0])/2**self.nlvl
        for it in range(0,self.out_filter_siz//2):
            KVal = np.exp(-2*math.pi*1j*xcen*(kNodes[it]-ChebNodes)*klen)
            LVec = np.squeeze(LMat[it,:])
            mat[0,range(0,4*NG,4),2*it  ] =   np.multiply(KVal.real,LVec)
            mat[0,range(1,4*NG,4),2*it  ] = - np.multiply(KVal.imag,LVec)
            mat[0,range(2,4*NG,4),2*it  ] = - np.multiply(KVal.real,LVec)
            mat[0,range(3,4*NG,4),2*it  ] =   np.multiply(KVal.imag,LVec)
            mat[0,range(0,4*NG,4),2*it+1] = - np.multiply(KVal.imag,LVec)
            mat[0,range(1,4*NG,4),2*it+1] = - np.multiply(KVal.real,LVec)
            mat[0,range(2,4*NG,4),2*it+1] =   np.multiply(KVal.imag,LVec)
            mat[0,range(3,4*NG,4),2*it+1] =   np.multiply(KVal.real,LVec)

        self.OutFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_Out" )
