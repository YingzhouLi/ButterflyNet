import numpy as np

from .dft1d_datasets import *

class Energy1D:
    def __init__(self, N, in_type = 'r', distfunc = lambda
            siz:np.random.uniform(-1,1,size=siz),
            x_range = [], x_magfunc = [], k_range = [], k_magfunc = [],
            batch_siz = 0, dtype = np.float32,
            energy_weightfunc = lambda k:np.divide(1.0,np.square(k))):

        self.batch_siz = batch_siz
        self.N = N
        self.distfunc  = distfunc
        self.x_magfunc = x_magfunc
        self.k_magfunc = k_magfunc
        self.energy_weightfunc = energy_weightfunc
        self.dtype = dtype

        if (in_type.lower() == 'r'):
            io_type = 'r2c'
        else:
            io_type = 'c2c'

        self.dft = DFT1D(N, io_type, distfunc, x_range, x_magfunc,
                k_range, k_magfunc, batch_siz, dtype)

    def batch_size(self,batch_siz):
        self.batch_siz = batch_siz

    def gen_data(self,nsamp = -1):
        if nsamp < 0:
            nsamp = self.batch_siz
        xdata, y = self.dft.gen_data(nsamp)
        yreal,yimag = self.vecassign_r2c(y)
        ymag = np.absolute(yreal+1j*yimag)
        with np.errstate(divide='ignore'):
            w = self.energy_weightfunc(self.dft.kval)
        w[w==np.inf] = 0
        edata = np.sum(np.multiply(np.square(ymag),w),
                axis=1, keepdims=True)
        return xdata.astype(self.dtype),edata.astype(self.dtype)

    def vecassign_r2c(self,x):
        xsiz = np.size(x,1)
        xreal = x[:,range(0,xsiz,2)]
        ximag = x[:,range(1,xsiz,2)]
        return xreal, ximag
