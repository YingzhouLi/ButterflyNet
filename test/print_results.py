import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"..")
import tensorflow as tf
import json
import numpy as np
import time
import math
from itertools import product

from butterflynet import models
from butterflynet import datasets
from butterflynet.utils import gaussianfun

def round_sig(x, sig=3):
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

def fexp(f):
    return int(math.floor(math.log10(abs(round_sig(f))))) if f != 0 else 0

def fexpabs(f):
    return abs(fexp(f))

def fexpsign(f):
    if fexp(f) > 0:
        return '+'
    else:
        return '-'

def fman(f):
    return f/10**fexp(f)

def get_norms(save_path):
    with open(save_path+'/output.out', 'r') as f_read:
        for line in f_read.readlines():
            if 'norm1' in line:
                break
    norm1 = float(line.split()[1])
    with open(save_path+'/output.out', 'r') as f_read:
        for line in f_read.readlines():
            if 'norm2' in line:
                break
    norm2 = float(line.split()[1])
    with open(save_path+'/output.out', 'r') as f_read:
        for line in f_read.readlines():
            if 'norminf' in line:
                break
    norminf = float(line.split()[1])
    return norm1, norm2, norminf

def get_numpara(save_path):
    with open(save_path+'/output.out', 'r') as f_read:
        for line in f_read.readlines():
            if 'Total number of parameters' in line:
                break
    return int(line.split()[4])

def get_numpara_task(save_path):
    with open(save_path+'/output.out', 'r') as f_read:
        for line in f_read.readlines():
            if 'Task              :' in line:
                break
    return int(line.split()[2])

def get_numpara_bnet(save_path):
    return get_numpara(save_path) - get_numpara_task(save_path)

def get_pretrainrelerr(save_path):
    with open(save_path+'/output.out', 'r') as f_read:
        for line in f_read.readlines():
            if 'Iter        0:' in line:
                break
    line = line.replace(',', ' ')
    return float(line.split()[5])

def get_testrelerr(save_path):
    with open(save_path+'/output.out', 'r') as f_read:
        last_line = f_read.readlines()[-1]
    last_line = last_line.replace(',', ' ')
    return float(last_line.split()[6])


freq_list     = [64,256]
Lk_list       = range(1,4)

for itL in range(0,3):
    print("\\midrule")
    for itLk in range(0,len(Lk_list)):
        Lk = Lk_list[itLk]
        for freq_max in freq_list:
            Lmax = int(np.log2(freq_max))
            L_list = range(Lmax-2, Lmax+1)
            L = L_list[itL]

            freqstr = 'freq'+str(freq_max)
            Lstr    = 'L'+str(L)
            Lkstr   = 'Lk'+str(Lk)
            save_path = 'approx' \
                    + '/' + freqstr \
                    + '/' + Lstr \
                    + '/' + Lkstr

            n1, n2, ninf = get_norms(save_path)

            if freq_max != freq_list[0]:
                print("&")

            if Lk == 1:
                print("\\multirow{3}{*}{%d}" % L)

            print(("& %2d & %4.2f\\np{e%s}%d & %4.2f\\np{e%s}%d " \
                    +"& %4.2f\\np{e%s}%d ") \
                    % (Lk, fman(n1), fexpsign(n1), fexpabs(n1), \
                       fman(n2), fexpsign(n2), fexpabs(n2), \
                       fman(ninf), fexpsign(ninf), fexpabs(ninf)))

        print("\\\\")

print("========================================================")

ds_list     = ['dft', 'dftsmooth']
freq_list   = ['low_freq', 'high_freq']
Lk_list     = ['Lk1', 'Lk2', 'Lk3']
nn_list     = ['bnet', 'ibnet']
prefix_list = ['prefix', 'random']

print("\\toprule")
for itLk in range(0,len(Lk_list)):
    Lk = Lk_list[itLk]
    for nn, prefix in product(nn_list, prefix_list):
        if prefix == 'prefix':
            if nn == 'bnet':
                print(("\\multirow{4}{*}{%d} &"
                    + "\\multirow{2}{*}{\\sNetName} & prefix &") \
                    % (itLk+1))
            else:
                print("& \\multirow{2}{*}{\\sINetName} & prefix &")
        else:
            print("& & random &")
        pretrain = []
        test     = []
        for ds,freq in product(ds_list, freq_list):
            save_path = ds + '/' + freq + '/' + nn + '/channel_size16/' \
                    + Lk + '/' + prefix
            npara    = get_numpara(save_path)
            pretrain.append(get_pretrainrelerr(save_path))
            test.append(get_testrelerr(save_path))
        print("%8d " % npara)
        for it in range(len(test)):
            print("& %4.2f\\np{e%s}%d & %4.2f\\np{e%s}%d " \
                    % (fman(pretrain[it]), fexpsign(pretrain[it]), \
                    fexpabs(pretrain[it]), fman(test[it]), \
                    fexpsign(test[it]), fexpabs(test[it])))
        print("\\\\")
    if Lk != 'Lk3':
        print("\midrule")

print("========================================================")

ds_list     = ['energy', 'energyhighfreq']
task_list   = ['sqr', 'den']
Lk_list     = ['Lk1', 'Lk2', 'Lk3']
nn_list     = ['bnet', 'ibnet']
prefix_list = ['prefix', 'random']

for task in task_list:
    print("\\toprule")
    if task == 'sqr':
        print("\\multirow{12}{*}{Square-sum-layer}")
    else:
        print("\\multirow{12}{*}{Dense-dense-layer}")
    for itLk in range(0,len(Lk_list)):
        Lk = Lk_list[itLk]
        for nn, prefix in product(nn_list, prefix_list):
            if prefix == 'prefix':
                if nn == 'bnet':
                    print(("& \\multirow{4}{*}{%d} &"
                        + "\\multirow{2}{*}{\\sNetName} & prefix &") \
                        % (itLk+1))
                else:
                    print("& & \\multirow{2}{*}{\\sINetName} & prefix &")
            else:
                print("& & & random &")
            pretrain = []
            test     = []
            for ds in ds_list:
                save_path = ds + '/' + task + '/' + nn + '/channel_size16/' \
                        + Lk + '/' + prefix
                ntaskpara    = get_numpara_task(save_path)
                nbnetpara    = get_numpara_bnet(save_path)
                pretrain.append(get_pretrainrelerr(save_path))
                test.append(get_testrelerr(save_path))
            print("%8d & %8d " % (nbnetpara, ntaskpara))
            for it in range(len(test)):
                print("& %4.2f\\np{e%s}%d & %4.2f\\np{e%s}%d " \
                        % (fman(pretrain[it]), fexpsign(pretrain[it]), \
                        fexpabs(pretrain[it]), fman(test[it]), \
                        fexpsign(test[it]), fexpabs(test[it])))
            print("\\\\")
        if Lk != 'Lk3':
            print("\cmidrule(lr){2-8}")
