import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
from pathlib import Path
import math
import numpy as np
import scipy.io as spio
import tensorflow as tf
from matplotlib import pyplot as plt

from gaussianfun import gaussianfun
from gen_dft_data import gen_dft_data
from ButterflyLayer import ButterflyLayer

N = 64
Ntest = 100000
in_siz = N # Length of input vector
out_siz = N//8*2 # Length of output vector
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])
freqidx = range(out_siz//2)
freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),[-5,0,5],[2,2,2]))

#=========================================================
#----- Parameters Setup

prefixed = False

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

testInData = tf.placeholder(tf.float32, shape=(1,in_siz,1),
        name="testInData")
testOutData = tf.placeholder(tf.float32, shape=(1,out_siz,1),
        name="testOutData")

#=========================================================
#----- Training Preparation
butterfly_net = ButterflyLayer(in_siz, out_siz,
        in_filter_siz, out_filter_siz,
        channel_siz, nlvl, prefixed,
        in_range, out_range)

y_output = butterfly_net(testInData)
loss_test = tf.reduce_mean(tf.squared_difference(testOutData, y_output))

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training
sess.run(init)

x_test_data_file = Path("./tftmp/x_test_data.npy")
y_test_data_file = Path("./tftmp/y_test_data.npy")
if x_test_data_file.exists() & y_test_data_file.exists():
    x_test_data = np.load(x_test_data_file)
    y_test_data = np.load(y_test_data_file)
else:
    x_test_data,y_test_data = gen_dft_data(freqmag,freqidx,Ntest)
    np.save(x_test_data_file,x_test_data)
    np.save(y_test_data_file,y_test_data)

hist_loss_test_prefix = []
test_y_prefix = []
for it in range(Ntest):
    rand_x = x_test_data[it,:,:].reshape((1,-1,1))
    rand_y = y_test_data[it,:,:].reshape((1,-1,1))
    test_dict = {testInData: rand_x, testOutData: rand_y}
    temp_test_loss = sess.run(loss_test,feed_dict=test_dict)
    hist_loss_test_prefix.append(temp_test_loss)
    temp_test_y_prefix = sess.run(y_output, feed_dict=test_dict)
    test_y_prefix.append(temp_test_y_prefix)
    print("Iter # %6d: Test Loss: %10e." % (it+1,temp_test_loss))
    sys.stdout.flush()

np.save("./tftmp/hist_loss_test_rand.npy",hist_loss_test_prefix)
np.save("./tftmp/test_y_rand.npy",test_y_prefix)

sess.close()
