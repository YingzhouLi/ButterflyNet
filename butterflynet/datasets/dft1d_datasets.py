import numpy as np

class DFT1D:
    def __init__(self, N, io_type = 'r2r', xdistfunc = lambda
            siz:np.random.uniform(size=siz),
            x_range = [], k_range = [], batch_siz = 0, dtype = np.float32):
        self.batch_siz = batch_siz
        self.N = N
        self.xdistfunc = xdistfunc
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
        for it in range(N):
            x = it/N
            if (x >= x_range[0]) and (x < x_range[1]):
                self.xidx.append(it)
        self.xlen = len(self.xidx)

        fftfreqs = np.fft.fftshift(np.fft.fftfreq(N))*N + 0.1
        self.kidx = []
        for it in range(N):
            k = fftfreqs[it]
            if (k >= k_range[0]) and (k < k_range[1]):
                self.kidx.append(it)
        self.klen = len(self.kidx)

    def batch_size(self,batch_siz):
        self.batch_siz = batch_siz

    def gen_data(self,nsamp = 0):
        if nsamp == 0:
            nsamp = self.batch_siz
        if self.xtype == 'r':
            if self.ktype == 'r':
                x,y = self.gen_data_r2r(nsamp)
            else:
                x,y = self.gen_data_r2c(nsamp)
        else:
            if self.ktype == 'r':
                x,y = self.gen_data_c2r(nsamp)
            else:
                x,y = self.gen_data_c2c(nsamp)
        return x.astype(self.dtype),y.astype(self.dtype)

    def gen_data_r2r(self,nsamp):
        xlong = np.zeros([nsamp,self.N])
        x = self.xdistfunc([nsamp,self.xlen])
        xlong[:,self.xidx] = x
        ylong = np.fft.fftshift(np.fft.fft(xlong),1)
        y = ylong.real[:,self.kidx]
        return x,y

    def gen_data_r2c(self,nsamp):
        xlong = np.zeros([nsamp,self.N])
        x = self.xdistfunc([nsamp,self.xlen])
        xlong[:,self.xidx] = x
        ylong = np.fft.fftshift(np.fft.fft(xlong),1)
        y = self.vecassign_c2r(ylong.real[:,self.kidx],
                ylong.imag[:,self.kidx])
        return x,y

    def gen_data_c2r(self,nsamp):
        xlong = np.zeros([nsamp,self.N],dtype=complex)
        xreal = self.xdistfunc([nsamp,self.xlen])
        ximag = self.xdistfunc([nsamp,self.xlen])
        x = self.vecassign_c2r(xreal,ximag)
        xlong[:,self.xidx] = xreal+1j*ximag
        ylong = np.fft.fftshift(np.fft.fft(xlong),1)
        y = ylong.real[:,self.kidx]
        return x,y

    def gen_data_c2c(self,nsamp):
        xlong = np.zeros([nsamp,self.N],dtype=complex)
        xreal = self.xdistfunc([nsamp,self.xlen])
        ximag = self.xdistfunc([nsamp,self.xlen])
        x = self.vecassign_c2r(xreal,ximag)
        xlong[:,self.xidx] = xreal+1j*ximag
        ylong = np.fft.fftshift(np.fft.fft(xlong),1)
        y = self.vecassign_c2r(ylong.real[:,self.kidx],
                ylong.imag[:,self.kidx])
        return x,y
    
    def vecassign_c2r(self,xreal,ximag):
        nsamp = np.size(xreal,0)
        xreal = np.reshape(xreal,(nsamp,1,-1),'F')
        ximag = np.reshape(ximag,(nsamp,1,-1),'F')
        return np.reshape(np.concatenate((xreal,ximag),1),(nsamp,-1),'F')
