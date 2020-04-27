import numpy as np

class Identity:
    def __init__(self, N, io_type = 'r2r',
            x_range = [], k_range = [],
            dtype = np.float32):

        self.N = N
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

    def gen_data(self):
        if self.xtype == 'r':
            xdata, x = self.gen_xdata_r()
        else:
            xdata, x = self.gen_xdata_c()
        if self.ktype == 'r':
            ydata = self.get_ydata_r(x)
        else:
            ydata = self.get_ydata_c(x)
        return xdata.astype(self.dtype),ydata.astype(self.dtype)

    def gen_xdata_r(self):
        x = np.eye(self.xlen)
        xdata = x
        return xdata, x

    def gen_xdata_c(self):
        xreal = np.eye(self.xlen)
        ximag = np.zeros((self.xlen, self.xlen))
        x = xreal+1j*ximag
        xdata = self.vecassign_c2r(xreal,ximag)
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
