import numpy as np
def gen_uni_data(freqmag,freqidx,siz):
    N = len(freqmag)

    freqmag = np.tile(np.reshape(freqmag,[1,N]),(siz,1))
    consty = np.random.uniform(-np.sqrt(30),np.sqrt(30),[siz,1])
    zeroy = np.zeros([siz,1])
    if N % 2 == 0:
        halfy = np.random.uniform(-np.sqrt(15),np.sqrt(15),[siz,N//2-1])
        realy = np.concatenate((consty,halfy,zeroy,halfy[:,::-1]),axis=1)
        halfy = np.random.uniform(-np.sqrt(15),np.sqrt(15),[siz,N//2-1])
        imagy = np.concatenate((zeroy,halfy,zeroy,-halfy[:,::-1]),axis=1)
    else:
        halfy = np.random.uniform(-np.sqrt(15),np.sqrt(15),[siz,N//2])
        realy = np.concatenate((consty,halfy,halfy[:,::-1]),axis=1)
        halfy = np.random.uniform(-np.sqrt(15),np.sqrt(15),[siz,N//2])
        imagy = np.concatenate((zeroy,halfy,-halfy[:,::-1]),axis=1)

    realy = realy*freqmag
    imagy = imagy*freqmag
    y = realy + imagy*1j
    xdata = np.reshape(np.fft.ifft(y,N,1).real,(siz,N,1),order='F')
    y = np.reshape(np.fft.fft(xdata,N,1),(siz,1,N),order='F')
    realy = y.real[:,:,freqidx]
    imagy = y.imag[:,:,freqidx]
    ydata = np.reshape(np.concatenate((realy,imagy),axis=1),(siz,-1,1),order='F')
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    return xdata,ydata

def gen_gaussian_data(freqmag,freqidx,siz):
    N = len(freqmag)
    freqmag = np.tile(np.reshape(freqmag,[1,N]),(siz,1))
    consty = np.random.uniform(-np.sqrt(2),np.sqrt(2),[siz,1])
    zeroy = np.zeros([siz,1])
    if N % 2 == 0:
        halfy = np.random.uniform(-1,1,[siz,N//2-1])
        realy = np.concatenate((consty,halfy,zeroy,halfy[:,::-1]),axis=1)
        halfy = np.random.uniform(-1,1,[siz,N//2-1])
        imagy = np.concatenate((zeroy,halfy,zeroy,-halfy[:,::-1]),axis=1)
    else:
        halfy = np.random.uniform(-1,1,[siz,N//2])
        realy = np.concatenate((consty,halfy,halfy[:,::-1]),axis=1)
        halfy = np.random.uniform(-1,1,[siz,N//2])
        imagy = np.concatenate((zeroy,halfy,-halfy[:,::-1]),axis=1)

    realy = realy*freqmag
    imagy = imagy*freqmag
    y = N*(realy + imagy*1j)
    xdata = np.reshape(np.fft.ifft(y,N,1).real,(siz,N,1),order='F')
    y = np.reshape(np.fft.fft(xdata,N,1),(siz,1,N),order='F')
    realy = y.real[:,:,freqidx]
    imagy = y.imag[:,:,freqidx]
    ydata = np.reshape(np.concatenate((realy,imagy),axis=1),(siz,-1,1),order='F')
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    return xdata,ydata

def gen_energy_data(N,siz):
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

def gen_energy2d_data(N,siz):
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
