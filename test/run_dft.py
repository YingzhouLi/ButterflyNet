import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"..")
import tensorflow as tf
import json
import numpy as np
import time
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
nn_type  = nnparas['neural network type']
N        = nnparas['N']
c_siz    = nnparas['channel size']
io_type  = nnparas.get('inout type','r2c')
Lx       = nnparas.get('num of layers before switch',-1)
Lk       = nnparas.get('num of layers after switch',-1)
prefixed = nnparas.get('prefixed', False)
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
print("prefixed:                     %30r" % (prefixed))
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
batch_siz   = ttparas.get('batch size', 128)
max_iter    = ttparas.get('max num of iteration', 1e4)
report_freq = ttparas.get('report frequency', 10)
train_algo  = ttparas.get('training algorithm','adam')

if train_algo.lower() == 'adam':
    taparas = ttparas.get('adam',[])
lr_type     = taparas.get('learning rate', 1e-3)
beta1       = taparas.get('beta1', 0.9)
beta2       = taparas.get('beta2', 0.999)

if not isinstance(lr_type, (float)):
    lrparas = ttparas.get(lr_type,[])
    if (lr_type.lower() == 'exponential decay') \
            or (lr_type.lower() == 'inverse time decay'):
        init_learn_rate = lrparas.get('initial learning rate', 1e-3)
        decay_steps     = lrparas.get('decay steps', 100)
        decay_rate      = lrparas.get('decay rate', 0.95)

save_path   = ttparas.get('save folder path', [])


print("================ Train and Test Parameters =================")
print("num of test data:             %30d" % (n_test))
print("batch size:                   %30d" % (batch_siz))
print("max num of iteration:         %30d" % (max_iter))
print("report frequency:             %30d" % (report_freq))
print("training algorithm:           %30s" % (train_algo))
if isinstance(lr_type, (float)):
    print("    learning rate:            %30.3e" % (lr_type))
else:
    print("    learning rate:            %30s" % (lr_type))
    if (lr_type.lower() == 'exponential decay') \
            or (lr_type.lower() == 'inverse time decay'):
        print("        initial learning rate:    %26.3e" % (init_learn_rate))
        print("        decay steps:              %26d" % (decay_steps))
        print("        decay rate:               %26.4f" % (decay_rate))

print("    beta1:                    %30.4f" % (beta1))
print("    beta2:                    %30.4f" % (beta2))
if save_path:
    print("save folder path:             %30s" % (save_path))
print("")


#=========================================================
#----- Network Preparation

if nn_type  == "bnet":
    model = models.ButterflyNet1D(in_siz, out_siz, io_type, c_siz,
            L, Lx, Lk, prefixed, x_range, k_range)
else:
    model = models.InflatedButterflyNet1D(in_siz, out_siz, io_type, c_siz,
            L, Lx, Lk, prefixed, x_range, k_range)

model.summary()

def magfunc(x):
    return gaussianfun(x, g_means, g_stds)
if ds_type.lower() == 'dft':
    dataset = datasets.DFT1D(N, io_type, x_range=x_range, k_range=k_range)
elif ds_type.lower() == 'dft gaussian smooth':
    dataset = datasets.DFT1D(N, io_type, k_magfunc = magfunc,
            x_range=x_range, k_range=k_range)
dataset.batch_size(batch_siz)

def compute_loss(y,ytrue):
    nfrac = tf.reduce_sum(tf.math.squared_difference(y,ytrue),axis=[1])
    dfrac = tf.reduce_sum(tf.square(ytrue), axis=[1])
    return tf.reduce_mean(tf.divide(nfrac,dfrac))

def compute_relative_error(y,ytrue):
    nfrac = tf.reduce_sum(tf.math.squared_difference(y,ytrue),axis=[1])
    dfrac = tf.reduce_sum(tf.square(ytrue), axis=[1])
    return tf.reduce_mean(tf.sqrt(tf.divide(nfrac,dfrac)))


if isinstance(lr_type, (float)):
    learn_rate = lr_type
else:
    if (lr_type.lower() == 'exponential decay'):
        learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate = init_learn_rate,
                decay_steps = decay_steps, decay_rate = decay_rate)
    elif (lr_type.lower() == 'inverse time decay'):
        learn_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate = init_learn_rate,
                decay_steps = decay_steps, decay_rate = decay_rate)

optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate,
        beta_1=beta1, beta_2=beta2)


#=========================================================
#----- Step by Step Training
@tf.function
def train_one_step(model, optimizer, x, ytrue):
    with tf.GradientTape() as tape:
        y = model(x)
        loss = compute_loss(y,ytrue)
        relerr = compute_relative_error(y,ytrue)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, relerr

def train(model, optimizer, dataset):
    loss = 0.0
    starttim = time.time()
    losshist = []
    relerrhist = []
    for it in range(max_iter):
        x, ytrue = dataset.gen_data()
        loss, relerr = train_one_step(model, optimizer, x, ytrue)
        losshist.append(loss.numpy())
        relerrhist.append(relerr.numpy())
        if it % report_freq == 0:
            endtim = time.time()
            print(("Iter %8d:  loss %14.8e,  relerr: %14.8e,  " \
                    + "runtime: %8.2f") \
                    % (it, loss.numpy(), relerr.numpy(), endtim-starttim))
    endtim = time.time()
    print(("Iter %8d:  loss %14.8e,  relerr: %14.8e,  " \
            + "runtime: %8.2f") \
            % (it, loss.numpy(), relerr.numpy(), endtim-starttim))
    return losshist, relerrhist

losshist, relerrhist = train(model, optimizer, dataset)


if save_path:
    model.save_weights(save_path+'/model')
    np.savez_compressed(save_path+'/hists.npz', losshist=losshist,
            relerrhist=relerrhist)
    plt.semilogy(relerrhist)
    plt.xlabel('Iteration')
    plt.ylabel('Relative Error')
    plt.savefig(save_path+'/relerrhist.pdf')
    plt.clf()

x, ytrue = dataset.gen_data(n_test)
starttim = time.time()
y = model(x)
loss = compute_loss(y,ytrue)
relerr = compute_relative_error(y,ytrue)
endtim = time.time()
print(("Test Result  :  loss %14.8e,  relerr: %14.8e,  " \
        + "runtime: %8.2f") \
        % (loss.numpy(), relerr.numpy(), endtim-starttim))
