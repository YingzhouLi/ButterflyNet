import numpy as np

class DFT1D:
    def __init__(self, N, io_type = 'r2r', distfunc = lambda
            siz:np.random.uniform(-1,1,size=siz),
            x_range = [], x_magfunc = [], k_range = [], k_magfunc = [],
            batch_siz = 0, dtype = np.float32):
        self.batch_siz = batch_siz
        self.N = N
        self.distfunc  = distfunc
        self.x_magfunc = x_magfunc
        self.k_magfunc = k_magfunc
        self.dtype = dtype

        if (io_type.lower() == 'r2r') \
                or (io_type.lower() == 'r2c'):
            self.xtype = 'r'
        else:
            self.xtype = 'c'
        if (io_type.lower() == 'r2r') \
                or (io_type.lower() == 'c2r'):
            self.ktype = 'r'
        else:
            self.ktype = 'c'

        if not x_range:
            x_range = [0,1]
        if not k_range:
            k_range = [-N/2, N/2]

        self.xidx = []
        self.xval = []
        for it in range(N):
            x = it/N
            if (x >= x_range[0]) and (x < x_range[1]):
                self.xidx.append(it)
                self.xval.append(x)
        self.xlen = len(self.xidx)

        fftfreqs = np.fft.fftshift(np.fft.fftfreq(N))*N + 0.1
        self.kidx = []
        self.kval = []
        for it in range(N):
            k = fftfreqs[it]
            if (k >= k_range[0]) and (k < k_range[1]):
                self.kidx.append(it)
                self.kval.append(k-0.1)
        self.klen = len(self.kidx)

    def batch_size(self,batch_siz):
        self.batch_siz = batch_siz

    def gen_data(self,nsamp = -1):
        if nsamp < 0:
            nsamp = self.batch_siz
        if self.xtype == 'r':
            xdata, x = self.gen_xdata_r(nsamp)
        else:
            xdata, x = self.gen_xdata_c(nsamp)
        if self.ktype == 'r':
            ydata = self.get_ydata_r(x)
        else:
            ydata = self.get_ydata_c(x)
        return xdata.astype(self.dtype),ydata.astype(self.dtype)

    def gen_xdata_r(self,nsamp):
        if (not self.x_magfunc) and (not self.k_magfunc):
            x = self.distfunc([nsamp,self.xlen])
            xdata = x
        elif (not self.k_magfunc):
            x = np.multiply(self.distfunc([nsamp,self.xlen]),
                    self.x_magfunc(self.xval))
            xdata = x
        else:
            magvec = self.k_magfunc(np.arange(-self.N/2,self.N/2,1))
            yreal = np.multiply(self.distfunc([nsamp,self.N]),
                        magvec)
            yimag = np.multiply(self.distfunc([nsamp,self.N]),
                        magvec)
            ylong = yreal+1j*yimag
            xlong = np.fft.ifft(np.fft.fftshift(ylong,1))
            x = xlong.real[:,self.xidx]
            xdata = x

        return xdata, x

    def gen_xdata_c(self,nsamp):
        if (not self.x_magfunc) and (not self.k_magfunc):
            xreal = self.distfunc([nsamp,self.xlen])
            ximag = self.distfunc([nsamp,self.xlen])
            x = xreal+1j*ximag
            xdata = self.vecassign_c2r(xreal,ximag)
        elif (not self.k_magfunc):
            magvec = self.x_magfunc(self.xval)
            xreal = np.multiply(self.distfunc([nsamp,self.xlen]),
                    magvec)
            ximag = np.multiply(self.distfunc([nsamp,self.xlen]),
                    magvec)
            x = xreal+1j*ximag
            xdata = self.vecassign_c2r(xreal,ximag)
        else:
            magvec = self.k_magfunc(np.arange(-self.N/2,self.N/2,1))
            yreal = np.multiply(self.distfunc([nsamp,self.N]),
                        magvec)
            yimag = np.multiply(self.distfunc([nsamp,self.N]),
                        magvec)
            ylong = yreal+1j*yimag
            xlong = np.fft.ifft(np.fft.fftshift(ylong,1))
            x = xlong[:,self.xidx]
            xdata = self.vecassign_c2r(xlong[:,self.xidx].real,
                    xlong[:,self.xidx].imag)
        return xdata, x

    def get_ydata_r(self,x):
        xlong = np.zeros([x.shape[0],self.N])
        xlong[:,self.xidx] = x
        ylong = np.fft.fftshift(np.fft.fft(xlong),1)
        y = ylong.real[:,self.kidx]
        return y

    def get_ydata_c(self,x):
        xlong = np.zeros([x.shape[0],self.N],dtype=complex)
        xlong[:,self.xidx] = x
        ylong = np.fft.fftshift(np.fft.fft(xlong),1)
        y = self.vecassign_c2r(ylong.real[:,self.kidx],
                ylong.imag[:,self.kidx])
        return y

    def vecassign_c2r(self,xreal,ximag):
        nsamp = np.size(xreal,0)
        xreal = np.reshape(xreal,(nsamp,1,-1),'F')
        ximag = np.reshape(ximag,(nsamp,1,-1),'F')
        return np.reshape(np.concatenate((xreal,ximag),1),(nsamp,-1),'F')
