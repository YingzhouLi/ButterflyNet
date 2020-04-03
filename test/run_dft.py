import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"..")
import tensorflow as tf
import json
import numpy as np
import time

from butterflynet import models
from butterflynet import datasets

json_file = open(sys.argv[1])
paras = json.load(json_file)

nnparas = paras['neural network']
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

print("=========== Neural Network Parameters ==============")
print("neural network type:      %10s" % (nn_type))
print("N:                            %6d" % (N))
print("input size:                   %6d" % (in_siz))
print("output size:                  %6d" % (out_siz))
print("channel size:                 %6d" % (c_siz))
print("inout type:                   %6s" % (io_type))
print("num of layers:                %6d" % (L))
print("num of layers before switch:  %6d" % (Lx))
print("num of layers after switch:   %6d" % (Lk))
print("prefixed:                     %6r" % (prefixed))
print("input range:                  [%6.2f, %6.2f)" \
        % (x_range[0], x_range[1]))
print("output range:                 [%6.2f, %6.2f)" \
        % (k_range[0], k_range[1]))
print("")

#=========================================================
#----- Train and Test Parameters Setup
n_test      = ttparas.get('num of test data', 1000)
batch_siz   = ttparas.get('batch size', 128)
max_iter    = ttparas.get('max num of iteration', 1e4)
report_freq = ttparas.get('report frequency', 10)
ds_type     = ttparas.get('data set type','dft')
train_algo  = ttparas.get('training algorithm','adam')

train_algo  = train_algo.lower()
if train_algo == 'adam':
    taparas = ttparas.get('adam',[])
learn_rate  = taparas.get('learning rate', 1e-3)
decay_rate  = taparas.get('decay rate', [])
beta1       = taparas.get('beta1', 0.9)
beta2       = taparas.get('beta2', 0.999)

save_path   = ttparas.get('save model path', "")

print("=========== Train and Test Parameters ==============")
print("num of test data:             %6d" % (n_test))
print("batch size:                   %6d" % (batch_siz))
print("max num of iteration:         %6d" % (max_iter))
print("report frequency:             %6d" % (report_freq))
print("data set type:                %6s" % (ds_type))
print("training algorithm:           %6s" % (train_algo))
print("    learning rate:            %6.2e" % (learn_rate))
if decay_rate:
    print("    decay rate:               %6.4f" % (decay_rate))
print("    beta1:                    %6.4f" % (beta1))
print("    beta2:                    %6.4f" % (beta2))
print("save model path:              %6s" % (save_path))
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

def compute_loss(y,ytrue):
    nfrac = tf.reduce_sum(tf.math.squared_difference(y,ytrue),axis=[1])
    dfrac = tf.reduce_sum(tf.square(ytrue), axis=[1])
    return tf.reduce_mean(tf.divide(nfrac,dfrac))

def compute_relative_error(y,ytrue):
    nfrac = tf.reduce_sum(tf.math.squared_difference(y,ytrue),axis=[1])
    dfrac = tf.reduce_sum(tf.square(ytrue), axis=[1])
    return tf.reduce_mean(tf.sqrt(tf.divide(nfrac,dfrac)))

def xdistfunc(siz):
    return np.random.uniform(-1,1,size=siz)
if ds_type.lower() == 'dft':
    if (io_type.lower() == 'r2r') or (io_type.lower() == 'r2c'):
        N = int(in_siz/(x_range[1]-x_range[0]))
    else:
        N = int(in_siz/2/(x_range[1]-x_range[0]))
    dataset = datasets.DFT1D(N, io_type, xdistfunc,
            x_range=x_range, k_range=k_range)
dataset.batch_size(batch_siz)

if decay_rate:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate,
        beta_1=beta1, beta_2=beta2, decay_rate=decay_rate)
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate,
        beta_1=beta1, beta_2=beta2)


#=========================================================
#----- Step by Step Training
@tf.function
def train_one_step(model, optimizer, x, ytrue):
    with tf.GradientTape() as tape:
        y = model(x)
        loss = compute_loss(y,ytrue)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, dataset):
    loss = 0.0
    starttim = time.time()
    for it in range(max_iter):
        x, ytrue = dataset.gen_data()
        loss = train_one_step(model, optimizer, x, ytrue)
        if it % report_freq == 0:
            endtim = time.time()
            print("Iter %6d:  loss %14.8e,  time elapsed %8.2f" \
                    % (it, loss.numpy(), endtim-starttim))

train(model, optimizer, dataset)

if save_path != "":
    model.save_weights(save_path)

x, ytrue = dataset.gen_data(n_test)
y = model(x)
loss = compute_loss(y,ytrue)
relerr = compute_relative_error(y,ytrue)
print("Testing Result :  loss %14.8e,  relative error: %14.8e" \
                    % (loss.numpy(), relerr.numpy()))
