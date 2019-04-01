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
