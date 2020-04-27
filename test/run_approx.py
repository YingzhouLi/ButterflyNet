import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"..")
import tensorflow as tf
import json
import numpy as np
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from butterflynet import models
from butterflynet import datasets

json_file = open(sys.argv[1])
paras = json.load(json_file)

nnparas = paras['neural network']

#=========================================================
#----- Neural Network Parameters Setup
nn_type  = nnparas['neural network type']
N        = nnparas['N']
c_siz    = nnparas['channel size']
io_type  = nnparas.get('inout type','r2c')
Lx       = nnparas.get('num of layers before switch',-1)
Lk       = nnparas.get('num of layers after switch',-1)
init     = nnparas.get('initializer', 'glorot_uniform')
x_range  = nnparas.get('input range',[])
k_range  = nnparas.get('output range',[])

if (io_type.lower() == 'r2r') or (io_type.lower() == 'r2c'):
    in_siz = N*(x_range[1]-x_range[0])
else:
    in_siz = 2*N*(x_range[1]-x_range[0])
if (io_type.lower() == 'r2r') or (io_type.lower() == 'c2r'):
    out_siz = k_range[1] - k_range[0]
else:
    out_siz = 2*(k_range[1] - k_range[0])
L = Lx+Lk

print("================ Neural Network Parameters =================")
print("neural network type:          %30s" % (nn_type))
print("N:                            %30d" % (N))
print("input size:                   %30d" % (in_siz))
print("output size:                  %30d" % (out_siz))
print("channel size:                 %30d" % (c_siz))
print("inout type:                   %30s" % (io_type))
print("num of layers:                %30d" % (L))
print("num of layers before switch:  %30d" % (Lx))
print("num of layers after switch:   %30d" % (Lk))
print("initializer:                  %30s" % (init))
print("input range:                        [%10.3f, %10.3f)" \
        % (x_range[0], x_range[1]))
print("output range:                       [%10.3f, %10.3f)" \
        % (k_range[0], k_range[1]))
print("")


#=========================================================
#----- Network Preparation

if nn_type  == "bnet":
    model = models.ButterflyNet1D(in_siz, out_siz, io_type, c_siz,
            L, Lx, Lk, init, x_range, k_range)
else:
    model = models.InflatedButterflyNet1D(in_siz, out_siz, io_type, c_siz,
            L, Lx, Lk, init, x_range, k_range)

model.summary()

dataset = datasets.Identity(N, io_type, x_range=x_range, k_range=k_range)

def matassign_r2c(x):
    xsiz = tf.shape(x)[1].numpy()
    xreal = x[:, 0:xsiz:2]
    ximag = x[:, 1:xsiz:2]
    return tf.complex(xreal, ximag)

def compute_relative_norm(y,ytrue,p):
    ycmplx     = matassign_r2c(y)
    ytruecmplx = matassign_r2c(ytrue)
    nfrac = tf.norm(ycmplx-ytruecmplx, ord=p, axis=[0,1])
    dfrac = tf.norm(ytruecmplx, ord=p, axis=[0,1])
    return tf.divide(nfrac,dfrac)

x, ytrue = dataset.gen_data()

y = model(x)
norm1   = compute_relative_norm(y,ytrue,1)
norm2   = compute_relative_norm(y,ytrue,2)
norminf = compute_relative_norm(y,ytrue,np.inf)
print(("Test Result\n    norm1    %14.8e\n    norm2    %14.8e\n    " \
        + "norminf  %14.8e") \
        % (norm1.numpy(), norm2.numpy(), norminf.numpy()))
