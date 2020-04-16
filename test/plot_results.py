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

save_path_list = [
        'dftsmooth/low_freq/bnet/channel_size16/Lk3/prefix',
        'dftsmooth/low_freq/ibnet/channel_size16/Lk3/prefix',
        'dftsmooth/low_freq/bnet/channel_size16/Lk3/random',
        'dftsmooth/low_freq/ibnet/channel_size16/Lk3/random'
        ]

for save_path in save_path_list:
    loaded = np.load(save_path+'/trans_hists.npz')
    gmeanshist = np.array(loaded['gmeanshist'])
    relerrhist = np.array(loaded['relerrhist'])
    relstdhist = np.array(loaded['relstdhist'])
    plt.errorbar(gmeanshist, relerrhist, yerr=relstdhist, fmt='-')
    plt.yscale('log', nonposy='clip')
    plt.xlabel('Gaussian Mean')
    plt.ylabel('Relative Error')
plt.legend(['BNet-prefix', 'IBNet-prefix', 'BNet-random', 'IBNet-random'])
plt.savefig('trans.pdf')

