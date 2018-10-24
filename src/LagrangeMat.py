import numpy as np

def LagrangeMat(gs,ts): # gs and ts are vectors
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
