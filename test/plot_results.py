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
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import EngFormatter
from itertools import product

from butterflynet import models
from butterflynet import datasets
from butterflynet.utils import gaussianfun


def get_testrelerr(save_path):
    with open(save_path+'/output.out', 'r') as f_read:
        last_line = f_read.readlines()[-1]
    last_line = last_line.replace(',', ' ')
    return float(last_line.split()[6])


figure_path = 'figures'
plt.rcParams.update({'font.size': 18})

Lk_list = ['Lk1', 'Lk2', 'Lk3']

for Lk in Lk_list:
    save_path_list = [
            'dftsmooth/low_freq/bnet/channel_size16/'+Lk+'/prefix',
            'dftsmooth/low_freq/ibnet/channel_size16/'+Lk+'/prefix',
            'dftsmooth/low_freq/bnet/channel_size16/'+Lk+'/random',
            'dftsmooth/low_freq/ibnet/channel_size16/'+Lk+'/random'
            ]
    for save_path in save_path_list:
        loaded = np.load(save_path+'/trans_hists.npz')
        gmeanshist = np.array(loaded['gmeanshist'])
        relerrhist = np.array(loaded['relerrhist'])
        relstdhist = np.array(loaded['relstdhist'])
        plt.errorbar(gmeanshist, relerrhist, yerr=relstdhist, fmt='-')
        plt.yscale('log', nonposy='clip')
        plt.ylim([8e-6, 1.8])
        plt.xlabel('Gaussian Mean')
        plt.ylabel('Relative Error')
    plt.legend(['BNet-prefix', 'IBNet-prefix',
        'BNet-random', 'IBNet-random'])
    plt.tight_layout()
    plt.savefig(figure_path+'/trans'+Lk+'.pdf')
    plt.clf()

nn_list = ['bnet', 'ibnet']
prefix_list = ['prefix', 'random']
for nn in nn_list:
    for prefix in prefix_list:
        save_path = 'dftsmooth/low_freq/' + nn \
                + '/channel_size16/Lk1/' + prefix
        loaded = np.load(save_path+'/hists.npz')
        relerrhist = np.array(loaded['relerrhist'])
        plt.semilogy(relerrhist)
        if prefix == 'prefix':
            plt.ylim([8e-6, 1e-1])
        else:
            plt.ylim([8e-3, 1.8])
        plt.xlabel('Iteration')
        plt.ylabel('Relative Error')
        plt.gca().xaxis.set_major_formatter(EngFormatter())
        plt.tight_layout()
        plt.savefig(figure_path + '/conv-dftsmooth-low_freq-' \
                + nn + '-Lk1-'+prefix+'.pdf')
        plt.clf()


ds_list     = ['dft', 'dftsmooth']
freq_list   = ['low_freq', 'high_freq']
Lk_list     = ['Lk1', 'Lk2', 'Lk3']
nn_list     = ['bnet', 'ibnet']
prefix_list = ['prefix', 'random']

for ds, freq in product(ds_list, freq_list):
    for prefix, nn in product(prefix_list, nn_list):
        testerr = []
        for Lk in Lk_list:
            save_path = ds + '/' + freq + '/' + nn + '/channel_size16/' \
                    + Lk + '/' + prefix
            testerr.append(get_testrelerr(save_path))
        plt.semilogy(range(1,4,1), testerr)
    plt.ylim([8e-6, 1.8])
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('$L_\\xi$')
    plt.ylabel('Relative Error')
    plt.legend(['BNet-prefix', 'IBNet-prefix',
        'BNet-random', 'IBNet-random'])
    plt.tight_layout()
    plt.savefig(figure_path+'/testerr-'+ds+'-'+freq+'.pdf')
    plt.clf()

ds_list     = ['dft', 'dftsmooth']
freq_list   = ['low_freq', 'high_freq']
os.makedirs('_tmp',  exist_ok=True)
for ds, freq in product(ds_list, freq_list):
    json_file = ds + '/' + freq + \
        '/bnet/channel_size16/Lk1/prefix/para.json'
    os.system('python3 plot_data.py '+json_file+' _tmp/')
    os.system('cp _tmp/xreal.pdf ' + figure_path \
            + '/xreal-' + ds + '-' + freq + '.pdf')
    os.system('cp _tmp/yreal.pdf ' + figure_path \
            + '/yreal-' + ds + '-' + freq + '.pdf')
