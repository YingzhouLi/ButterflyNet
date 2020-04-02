import numpy as np

def gen_energy1d(N,siz):
    DW = np.fft.fftfreq(N)
    DW[DW==0] = np.inf
    DW = 1/DW
    tmp = np.random.normal(0,1,[siz,N//8])
    xdata = np.fft.irfft(np.fft.rfft(tmp,axis=1),N,1)
    ydata = np.sum(np.absolute(np.multiply(
        np.fft.fft(xdata,axis=1), DW))**2,axis=1)/N**2
    xdata = np.float32(np.reshape(xdata,[siz,N,1]))
    ydata = np.float32(np.reshape(ydata,[siz,1,1]))
    return xdata,ydata

def gen_energy2d(N,siz):
    DW = np.fft.fftfreq(N)
    DW[DW==0] = np.inf
    DW = 1/DW
    DW2 = np.outer(DW,DW)
    tmp = np.random.normal(0,1,[siz,N//8,N//8])
    xdata = np.fft.irfft(np.fft.rfft(tmp,axis=1),N,1)
    xdata = np.fft.irfft(np.fft.rfft(xdata,axis=2),N,2)
    ydata = np.sum(np.absolute(np.multiply(
        np.fft.fft2(xdata,axes=(1,2)), DW2))**2,axis=(1,2))/N**4
    xdata = np.float32(np.reshape(xdata,[siz,N,N,1]))
    ydata = np.float32(np.reshape(ydata,[siz,1,1,1]))
    return xdata,ydata
