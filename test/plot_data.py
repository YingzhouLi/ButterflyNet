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

json_file = open(sys.argv[1])
paras = json.load(json_file)

nnparas = paras['neural network']
dsparas = paras['data set']
ttparas = paras['train and test']

#=========================================================
#----- Neural Network Parameters Setup
N        = nnparas['N']
io_type  = nnparas.get('inout type','r2c')
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

print("================ Neural Network Parameters =================")
print("N:                            %30d" % (N))
print("input size:                   %30d" % (in_siz))
print("output size:                  %30d" % (out_siz))
print("inout type:                   %30s" % (io_type))
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

save_path   = ttparas.get('save folder path', [])

def magfunc(x):
    return out_siz/2*math.sqrt(N)*gaussianfun(x, g_means, g_stds)
if ds_type.lower() == 'dft':
    dataset = datasets.DFT1D(N, io_type, x_range=x_range, k_range=k_range)
elif ds_type.lower() == 'dft gaussian smooth':
    dataset = datasets.DFT1D(N, io_type, k_magfunc = magfunc,
            x_range=x_range, k_range=k_range)

if save_path:
    x, ytrue = dataset.gen_data(1)
    if (io_type.lower() == 'r2r') or (io_type.lower() == 'r2c'):
        plt.plot(x[0])
        plt.savefig(save_path+'/xreal.pdf')
        plt.clf()
    else:
        plt.plot(x[0][range(0,in_siz,2)])
        plt.savefig(save_path+'/xreal.pdf')
        plt.clf()
        plt.plot(x[0][range(1,in_siz,2)])
        plt.savefig(save_path+'/ximag.pdf')
        plt.clf()

    if (io_type.lower() == 'r2r') or (io_type.lower() == 'c2r'):
        plt.plot(ytrue[0])
        plt.savefig(save_path+'/yreal.pdf')
        plt.clf()
    else:
        plt.plot(ytrue[0][range(0,out_siz,2)])
        plt.savefig(save_path+'/yreal.pdf')
        plt.clf()
        plt.plot(ytrue[0][range(1,out_siz,2)])
        plt.savefig(save_path+'/yimag.pdf')
        plt.clf()

