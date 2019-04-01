import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../../src")
sys.path.insert(0,"../../src/data_gen")
from pathlib import Path
import math
import numpy as np
import scipy.io as spio
import tensorflow as tf

from gen_dft_data import gen_gaussian_data
from ButterflyLayer import ButterflyLayer

N = 64
Ntrain = 100
in_siz = N # Length of input vector
out_siz = N//8*2 # Length of output vector
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])
freqidx = range(out_siz//2)
stepfun = np.zeros(N)
stepfun[N//2-8:N//2+8] = 1/8
freqmag = np.fft.ifftshift(stepfun)

x_train,y_train = gen_gaussian_data(freqmag,freqidx,Ntrain)

#=========================================================
#----- Parameters Setup

prefixed = True

#----- Tunable Parameters of BNet
channel_siz = 8 # Num of interp pts on each dim

#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
nlvl = 2*math.floor(math.log(min(in_siz,out_siz//2),2)/2)
# Filter size for the input and output
in_filter_siz = in_siz//2**nlvl
out_filter_siz = out_siz//2**nlvl

print("======== Parameters =========")
print("Channel Size:     %6d" % (channel_siz))
print("Num Levels:       %6d" % (nlvl))
print("Prefix Coef:      %6r" % (prefixed))
print("In Range:        (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
print("Out Range:       (%6.2f, %6.2f)" % (out_range[0], out_range[1]))

#=========================================================
#----- Variable Preparation
sess = tf.Session()

trainInData = tf.placeholder(tf.float32, shape=(Ntrain,in_siz,1),
        name="trainInData")
trainOutData = tf.placeholder(tf.float32, shape=(Ntrain,out_siz,1),
        name="trainOutData")

#=========================================================
#----- Training Preparation
butterfly_net = ButterflyLayer(in_siz, out_siz,
        in_filter_siz, out_filter_siz,
        channel_siz, nlvl, prefixed,
        in_range, out_range)

y_train_output = butterfly_net(tf.convert_to_tensor(x_train))

loss_train = tf.reduce_mean(
        tf.squared_difference(y_train, y_train_output))

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training
sess.run(init)
train_dict = {trainInData: x_train, trainOutData: y_train}
train_loss = sess.run(loss_train,feed_dict=train_dict)
print("Train Loss: %10e." % (train_loss))

for n in tf.global_variables():
    np.save('tftmp/'+n.name.split(':')[0], n.eval(session=sess))
    print(n.name.split(':')[0] + ' saved')

sess.close()
