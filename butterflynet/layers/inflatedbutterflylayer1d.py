import math
import numpy as np
import tensorflow as tf

from ..utils import LagrangeMat
from ..utils import matassign_c2c

class InflatedButterflyLayer1D(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, in_siz, out_siz, io_type,
            channel_siz, nlvl = -1, nlvlx = -1, nlvlk = -1,
            prefixed = False, in_range = [], out_range = []):
        super(InflatedButterflyLayer1D, self).__init__()
        self.L  = nlvl
        self.Lx = nlvlx
        self.Lk = nlvlk
        if self.L < 0:
            self.L = self.Lx+self.Lk
        elif (self.Lx < 0) or (self.Lk < 0):
            self.Lx = (self.L+1)//2
            self.Lk = self.L//2

        self.c_siz  = channel_siz

        self.io_type = io_type
        if (self.io_type.lower() == "r2r") \
                or (self.io_type.lower() == "r2c"):
            self.itype         = "r"
            self.N             = in_siz
        else:
            self.itype         = "c"
            self.N             = in_siz//2
        if (self.io_type.lower() == "r2r") \
                or (self.io_type.lower() == "c2r"):
            self.otype         = "r"
            self.K             = out_siz
        else:
            self.otype         = "c"
            self.K             = out_siz//2

        self.x_filter_siz  = max(1, self.N//2**self.L)
        self.k_filter_siz  = max(1, self.K//2**self.L)

        totLk = min(int(np.log2(self.K)),self.L)
        totLx = min(int(np.log2(self.N)),self.L)
        self.Lx1 = totLk - self.Lk
        self.Lx2 = self.Lx - self.Lx1
        self.Lk4 = totLx - self.Lx
        self.Lk3 = self.Lk - self.Lk4

        self.x_range       = in_range
        self.k_range       = out_range

        if prefixed:
            self.buildDFT()
        else:
            self.buildRand()

    #==================================================================
    # Forward structure in the layer
    def call(self, in_data):
        n_data = tf.shape(in_data)[0]
        n_dim = tf.keras.backend.ndim(in_data)
        if n_dim == 2:
            in_data = tf.reshape(in_data, (n_data,-1,1))

        if self.itype == "r":
            XInterp = tf.nn.conv1d(input=in_data, filters=self.XFilterVar,
                stride=self.x_filter_siz, padding='VALID')
        else:
            XInterp = tf.nn.conv1d(input=in_data, filters=self.XFilterVar,
                stride=2*self.x_filter_siz, padding='VALID')
        XInterp = tf.nn.relu(tf.nn.bias_add(XInterp, self.XBiasVar))

        tfVars = []
        tfVars.append(XInterp)

        for lvl in range(1, self.Lx+1):
            Var = tf.nn.conv1d(input=tfVars[lvl-1],
                filters=self.FilterVars[lvl],
                stride=2, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                self.BiasVars[lvl]))
            tfVars.append(Var)

        # Middle level
        lvl = self.Lx
        tmpVarsk = tf.reshape([], (n_data, 2**self.Lx1, 0))
        for itx in range(0,2**self.Lk4):
            tmpVars = tf.reshape([], (n_data, 0, self.c_siz))
            for itk in range(0,2**self.Lx1):
                tmpVar = tfVars[lvl][:,itx,
                        itk*self.c_siz:(itk+1)*self.c_siz]
                tmpVar = tf.matmul(tmpVar,self.MidDenseVars[itk][itx])
                tmpVar = tf.nn.relu( tf.nn.bias_add(
                    tmpVar, self.MidBiasVars[itk][itx] ) )
                tmpVar = tf.reshape(tmpVar, (n_data,1,self.c_siz))
                tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
            tmpVarsk = tf.concat([tmpVarsk, tmpVars], axis=2)
        tfVars[lvl] = tmpVarsk

        for lvl in range(self.Lx+1,self.Lx+self.Lk3+1):
            Var = tf.nn.conv1d(input=tfVars[lvl-1],
                    filters=self.FilterVars[lvl],
                    stride=1, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl]))
            tmpVarsk = tf.reshape([], (n_data,2**(lvl-self.Lx2),0))
            for itx in range(0,2**self.Lk4):
                tmpVars = tf.reshape([], (n_data, 0, self.c_siz))
                for itk in range(0,2**(lvl-1-self.Lx2)):
                    tmpVar = Var[:, itk,
                            (2*itx)*self.c_siz : (2*itx+2)*self.c_siz ]
                    tmpVar = tf.reshape(tmpVar, (n_data,2,self.c_siz))
                    tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
                tmpVarsk = tf.concat([tmpVarsk, tmpVars], axis=2)
            tfVars.append(tmpVarsk)

        for lvl in range(self.Lx+self.Lk3+1,self.L+1):
            tmptfVars = tf.reshape([], (n_data, 2**(lvl-self.Lx2), 0))
            for itx in range(0,2**(self.L-lvl)):
                tmpVars = tf.reshape([], (n_data, 0, self.c_siz))
                for itk in range(0,2**(lvl-1-self.Lx2)):
                    tmpVar = tfVars[lvl-1][:, itk,
                        2*itx*self.c_siz:(2*itx+2)*self.c_siz]
                    tmpVar = tf.reshape(tmpVar, (n_data,2,self.c_siz))
                    tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
                tmptfVars = tf.concat([tmptfVars, tmpVars], axis=2)
            Var = tf.nn.conv1d(input=tmptfVars,
                    filters=self.FilterVars[lvl],
                    stride=2, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl]))
            tmpVarsk = tf.reshape([], (n_data,2**(lvl-self.Lx2),0))
            for itx in range(0,2**(self.L-lvl)):
                tmpVars = tf.reshape([], (n_data, 0, self.c_siz))
                for itk in range(0,2**(lvl-1-self.Lx2)):
                    tmpVar = Var[:, itk,
                            (2*itx)*self.c_siz : (2*itx+2)*self.c_siz ]
                    tmpVar = tf.reshape(tmpVar, (n_data,2,self.c_siz))
                    tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
                tmpVarsk = tf.concat([tmpVarsk, tmpVars], axis=2)
            tfVars.append(tmpVarsk)

        KInterp = tf.nn.conv1d(input=tfVars[self.L],
            filters=self.KFilterVar, stride=1, padding='VALID')

        if n_dim == 2:
            out_data = tf.reshape(KInterp,shape=(n_data,-1))
        else:
            out_data = tf.reshape(KInterp,shape=(n_data,-1,1))

        return(out_data)


    #==================================================================
    # Initialize variables in the layer
    def buildRand(self):
        if self.itype == "r":
            self.XFilterVar = tf.Variable( tf.random.normal(
                [self.x_filter_siz, 1, self.c_siz],stddev=0.3),
                name="Filter_In" )
        else:
            self.XFilterVar = tf.Variable( tf.random.normal(
                [2*self.x_filter_siz, 1, self.c_siz],stddev=0.3),
                name="Filter_In" )
        self.XBiasVar = tf.Variable( tf.zeros([self.c_siz]),
            name="Bias_In" )

        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append([])
        self.BiasVars.append([])
        for lvl in range(1, self.Lx1+1):
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable(
                    tf.random.normal([2,2**(lvl-1)*self.c_siz,
                        2**lvl*self.c_siz],stddev=0.3),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**lvl*self.c_siz]),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        for lvl in range(self.Lx1+1, self.Lx+1):
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable(
                    tf.random.normal([2,2**self.Lx1*self.c_siz,
                        2**self.Lx1*self.c_siz],stddev=0.3),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**self.Lx1*self.c_siz]),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        self.MidDenseVars = []
        self.MidBiasVars = []
        for itk in range(0,2**self.Lx1):
            tmpMidDenseVars = []
            tmpMidBiasVars = []
            for itx in range(0,2**self.Lk4):
                varLabel = "LVL_Mid_%04d_%04d" % (itk,itx)
                denseVar = tf.Variable(
                        tf.random.normal(
                        [self.c_siz,self.c_siz],stddev=0.3),
                        name="Dense_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.c_siz]),
                        name="Bias_"+varLabel )
                tmpMidDenseVars.append(denseVar)
                tmpMidBiasVars.append(biasVar)
            self.MidDenseVars.append(list(tmpMidDenseVars))
            self.MidBiasVars.append(list(tmpMidBiasVars))

        for lvl in range(self.Lx+1,self.Lx+self.Lk3+1):
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable(
                    tf.random.normal(
                    [1,2**(self.Lk4)*self.c_siz,
                    2**(self.Lk4+1)*self.c_siz],stddev=0.3),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**(self.Lk4+1)*self.c_siz]),
                name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        for lvl in range(self.Lx+self.Lk3+1,self.L+1):
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable(
                    tf.random.normal(
                    [2,2**(self.L-lvl)*self.c_siz,
                    2**(self.L-lvl+1)*self.c_siz],stddev=0.3),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**(self.L-lvl+1)*self.c_siz]),
                name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        if self.otype == "r":
            self.KFilterVar = tf.Variable( tf.random.normal(
                [1, self.c_siz, self.k_filter_siz],stddev=0.3),
                name="Filter_Out" )
        else:
            self.KFilterVar = tf.Variable( tf.random.normal(
                [1, self.c_siz, 2*self.k_filter_siz],stddev=0.3),
                name="Filter_Out" )


    #==================================================================
    # Initialize variables with coeffs in BF in the layer
    def buildDFT(self):
        NG = self.c_siz//4
        xran = self.x_range[1] - self.x_range[0]
        kran = self.k_range[1] - self.k_range[0]
        ChebNodes = (np.cos(np.arange(2*NG-1,0,-2)/2/NG*math.pi)+1)/2
        ChebNodes = ChebNodes[np.newaxis,:]

        xNodes = np.arange(0,1,1.0/self.x_filter_siz)
        xNodes = xNodes[np.newaxis,:]
        LMat = LagrangeMat(ChebNodes,xNodes)

        #----------------
        # Setup initial interpolation weights
        kcen = np.mean(self.k_range)
        xlen = xran/2**(self.L-self.Lk3)
        KVal = np.exp(-2*math.pi*1j*kcen*(xNodes.T-ChebNodes)*xlen)
        tmpmat = matassign_c2c(LMat*KVal)
        if self.itype == "r":
            mat = np.empty((self.x_filter_siz,1,self.c_siz))
            mat[:,0,:] = tmpmat[range(0,4*self.x_filter_siz,4),:]
        else:
            mat = np.empty((2*self.x_filter_siz,1,self.c_siz))
            mat[range(0,2*self.x_filter_siz,2),0,:] = \
                    tmpmat[range(0,4*self.x_filter_siz,4),:]
            mat[range(1,2*self.x_filter_siz,2),0,:] = \
                    tmpmat[range(1,4*self.x_filter_siz,4),:]

        self.XFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_In" )
        self.XBiasVar = tf.Variable( tf.zeros([self.c_siz]),
                name="Bias_In" )

        #----------------
        # Setup right factor interpolation weights
        x1Nodes = ChebNodes/2
        x2Nodes = ChebNodes/2 + 1/2
        LMat1 = LagrangeMat(ChebNodes,x1Nodes)
        LMat2 = LagrangeMat(ChebNodes,x2Nodes)

        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append([])
        self.BiasVars.append([])

        for lvl in range(1,self.Lx1+1):
            bigmatf = np.zeros((2,2**(lvl-1)*self.c_siz,2**lvl*self.c_siz))
            for itk in range(0,2**lvl):
                mat = np.empty((2, self.c_siz, self.c_siz))
                kcen = kran/2**lvl*(itk+0.5) + self.k_range[0]
                xlen = xran/2**(self.L-lvl-self.Lk3)
                KVal = np.exp(-2*math.pi*1j*kcen*(x1Nodes.T-ChebNodes)*xlen)
                mat[0,:,:] = matassign_c2c(LMat1*KVal)
                KVal = np.exp(-2*math.pi*1j*kcen*(x2Nodes.T-ChebNodes)*xlen)
                mat[1,:,:] = matassign_c2c(LMat2*KVal)
                offset1 = itk//2*self.c_siz
                offset2 = itk*self.c_siz
                bigmatf[:, offset1:offset1+self.c_siz,
                        offset2:offset2+self.c_siz] = mat
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**lvl*self.c_siz]),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        for lvl in range(self.Lx1+1, self.Lx+1):
            bigmatf = np.zeros((2,2**self.Lx1*self.c_siz,
                2**self.Lx1*self.c_siz))
            for itk in range(0,2**self.Lx1):
                mat = np.empty((2, self.c_siz, self.c_siz))
                kcen = kran/2**self.Lx1*(itk+0.5) \
                        + self.k_range[0]
                xlen = xran/2**(self.L-lvl-self.Lk3)
                KVal = np.exp(-2*math.pi*1j*kcen*(x1Nodes.T-ChebNodes)*xlen)
                mat[0,:,:] = matassign_c2c(LMat1*KVal)
                KVal = np.exp(-2*math.pi*1j*kcen*(x2Nodes.T-ChebNodes)*xlen)
                mat[1,:,:] = matassign_c2c(LMat2*KVal)
                offset1 = itk*self.c_siz
                offset2 = itk*self.c_siz
                bigmatf[:, offset1:offset1+self.c_siz,
                        offset2:offset2+self.c_siz] = mat
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**self.Lx1*self.c_siz]),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        #----------------
        # Setup weights for middle layer
        self.MidDenseVars = []
        self.MidBiasVars = []
        for itk in range(0,2**self.Lx1):
            tmpMidDenseVars = []
            tmpMidBiasVars = []
            for itx in range(0,2**self.Lk4):
                varLabel = "LVL_Mid_%04d_%04d" % (itk,itx)
                mat = np.empty((self.c_siz, self.c_siz))
                klen = kran/2**self.Lx1
                koff = klen*itk + self.k_range[0]
                kNodes = ChebNodes*klen + koff
                xlen = xran/2**self.Lk4
                xoff = xlen*itx + self.x_range[0]
                xNodes = ChebNodes*xlen + xoff
                KVal = np.exp(-2*math.pi*1j*np.outer(xNodes,kNodes))
                mat = matassign_c2c(KVal)
                denseVar = tf.Variable( mat.astype(np.float32),
                        name="Dense_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.c_siz]),
                        name="Bias_"+varLabel )
                tmpMidDenseVars.append(denseVar)
                tmpMidBiasVars.append(biasVar)
            self.MidDenseVars.append(list(tmpMidDenseVars))
            self.MidBiasVars.append(list(tmpMidBiasVars))

        #----------------
        # Setup left factor interpolation weights
        k1Nodes = ChebNodes/2
        k2Nodes = ChebNodes/2 + 1/2
        LMat1 = LagrangeMat(ChebNodes,k1Nodes)
        LMat2 = LagrangeMat(ChebNodes,k2Nodes)

        for lvl in range(self.Lx+1,self.Lx+self.Lk3+1):
            bigmatf = np.zeros((1, 2**(self.Lk4)*self.c_siz,
                    2**(self.Lk4+1)*self.c_siz))
            for itx in range(0,2**(self.Lk4+1)):
                varLabel = "LVL_%02d_%04d" % (lvl, itx)
                if itx % 2 == 0:
                    LMat = LMat1
                    kNodes = k1Nodes
                else:
                    LMat = LMat2
                    kNodes = k2Nodes
                mat = np.empty((1, self.c_siz, self.c_siz))
                klen = kran/2**(lvl-self.Lx2-1)
                xcen = xran/2**self.Lk4*(itx//2+0.5) \
                        + self.x_range[0]
                KVal = np.exp(-2*math.pi*1j*xcen*(kNodes-ChebNodes.T)*klen)
                mat[0,:,:] = matassign_c2c(KVal*LMat.T)
                offset1 = itx//2*self.c_siz
                offset2 = itx*self.c_siz
                bigmatf[:, offset1:offset1+self.c_siz,
                        offset2:offset2+self.c_siz] = mat
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**(self.Lk4+1)*self.c_siz]),
                name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        for lvl in range(self.Lx+self.Lk3+1,self.L+1):
            bigmatf = np.zeros((2, 2**(self.L-lvl)*self.c_siz,
                    2**(self.L-lvl+1)*self.c_siz))
            for itx in range(0,2**(self.L-lvl+1)):
                varLabel = "LVL_%02d_%04d" % (lvl, itx)
                if itx % 2 == 0:
                    LMat = LMat1
                    kNodes = k1Nodes
                else:
                    LMat = LMat2
                    kNodes = k2Nodes
                mat = np.empty((2, self.c_siz, self.c_siz))
                klen = kran/2**(lvl-self.Lx2-1)
                xcen = xran/2**(self.L-lvl+1) \
                        * (itx//2*2+0.5) \
                        + self.x_range[0]
                KVal = np.exp(-2*math.pi*1j*xcen*(kNodes-ChebNodes.T)*klen)
                mat[0,:,:] = matassign_c2c(KVal*LMat.T)
                xcen = xran/2**(self.L-lvl+1) \
                        * ((itx//2*2+1)+0.5) \
                        + self.x_range[0]
                KVal = np.exp(-2*math.pi*1j*xcen*(kNodes-ChebNodes.T)*klen)
                mat[1,:,:] = matassign_c2c(KVal*LMat.T)
                offset1 = itx//2*self.c_siz
                offset2 = itx*self.c_siz
                bigmatf[:, offset1:offset1+self.c_siz,
                        offset2:offset2+self.c_siz] = mat
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**(self.L-lvl+1)*self.c_siz]),
                name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        #----------------
        # Setup final interpolation weights
        kNodes = np.arange(0,1,1.0/self.k_filter_siz)
        kNodes = kNodes[np.newaxis,:]
        LMat = LagrangeMat(ChebNodes,kNodes)
        xcen = np.mean(self.x_range)
        klen = kran/2**(self.L-self.Lx2)
        KVal = np.exp(-2*math.pi*1j*xcen*(kNodes-ChebNodes.T)*klen)

        if self.otype == "r":
            mat = np.empty((1,self.c_siz,self.k_filter_siz))
            mat[0,:,:] = matassign_c2c(KVal*LMat.T) \
                    [:,range(0,4*self.k_filter_siz,4)]
        else:
            mat = np.empty((1,self.c_siz,2*self.k_filter_siz))
            tmpmat = matassign_c2c(KVal*LMat.T)
            for it in range(self.k_filter_siz):
                mat[0,:,2*it]   = tmpmat[:,4*it]
                mat[0,:,2*it+1] = tmpmat[:,4*it+1]

        self.KFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_Out" )
