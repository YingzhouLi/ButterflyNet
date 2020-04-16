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
from butterflynet.utils import gaussianfun

save_path = sys.argv[1]
json_file = open(save_path+'/para.json')
paras = json.load(json_file)

nnparas = paras['neural network']
dsparas = paras['data set']
ttparas = paras['train and test']

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
#----- Data Set Parameters Setup
ds_type     = dsparas.get('data set type','dft')
if ds_type.lower() == 'dft gaussian smooth':
    gaparas     = dsparas.get('dft gaussian smooth',[])
    g_means     = gaparas.get('gaussian means', [])
    g_stds      = gaparas.get('gaussian stds', [])

print("=================== Data Set Parameters ====================")
print("data set type:                %30s" % (ds_type))
if ds_type.lower() == 'dft gaussian smooth':
    print('\n'.join([ \
            ( "    gaussian mean %3d:        %30.3f\n" \
            + "    gaussian std  %3d:        %30.3f" ) \
            % (m, g_means[m], m, g_stds[m]) for m in range(len(g_means))]))
print("")

#=========================================================
#----- Train and Test Parameters Setup
n_test      = ttparas.get('num of test data', 1000)

print("================ Train and Test Parameters =================")
print("num of test data:             %30d" % (n_test))
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

model.load_weights(save_path+'/model')

def compute_relative_error(y,ytrue):
    nfrac = tf.reduce_sum(tf.math.squared_difference(y,ytrue),axis=[1])
    dfrac = tf.reduce_sum(tf.square(ytrue), axis=[1])
    rel   = tf.sqrt(tf.divide(nfrac,dfrac))
    mean  = tf.reduce_mean(rel)
    std   = tf.math.reduce_std(rel)
    return mean, std

def magfunc(x, g_mean):
    return out_siz*math.sqrt(N)*gaussianfun(x, [g_mean, -g_mean], g_stds)

gmeanshist = []
relerrhist = []
relstdhist = []
for g_mean in range(0,45,2):
    dataset = datasets.DFT1D(N, io_type,
            k_magfunc=lambda x:magfunc(x,g_mean),
            x_range=x_range, k_range=k_range)
    x, ytrue = dataset.gen_data(n_test)
    y = model(x)
    relerr, relstd = compute_relative_error(y,ytrue)
    print(("Test Result  : g_mean %14.8e,  relerr: %14.8e,  " \
            + "relstd: %14.8e") \
            % (g_mean, relerr.numpy(), relstd.numpy()))
    gmeanshist.append(g_mean)
    relerrhist.append(relerr.numpy())
    relstdhist.append(relstd.numpy())

np.savez_compressed(save_path+'/trans_hists.npz', gmeanshist=gmeanshist,
        relerrhist=relerrhist, relstdhist=relstdhist)
