import numpy as np
def gaussianfun(x, mulist, siglist):
    # Lengths of mulist and siglist are assumed to be the same
    len_list = len(mulist)
    gx = np.zeros(x.shape)
    for it in range(len_list):
        mu   = mulist[it]
        sig2 = siglist[it]*siglist[it]
        gx   = gx + np.exp(-np.power(x-mu,2.)/(2*sig2)) \
               / np.sqrt(2*np.pi*sig2) / len_list
    return gx

def LagrangeMat(gs,ts): # gs and ts are vectors
    gs = np.squeeze(gs)
    ts = np.squeeze(ts)
    NG = gs.shape[0]
    NT = ts.shape[0]
    mat = np.ones((NT,NG))
    for itG in range(0,NG):
        for itT in range(0,NT):
            for it in range(0,NG):
                if it != itG:
                    mat[itT,itG] = mat[itT,itG] * (ts[itT]-gs[it]) \
                            / (gs[itG]-gs[it])
    return mat
