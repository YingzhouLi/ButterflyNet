import sys
sys.path.insert(0,"../src")
import math
import numpy as np
import scipy.io as spio
import tensorflow as tf

from ButterflyLayer import ButterflyLayer


#=========================================================
# Read data from file
#---------------------------------------------------------
data_fname = 'data_DFT_smooth.mat'

mat = spio.loadmat(data_fname,squeeze_me=False)

n_train = int(mat['n_train'])
in_siz = int(mat['in_siz']) # Length of input vector
out_siz = int(mat['out_siz']) # Length of output vector
in_range = np.squeeze(np.float32(mat['in_range']))
out_range = np.squeeze(np.float32(mat['out_range']))
x_train = np.float32(mat['x_train'])
x_train = np.reshape(x_train,(n_train,in_siz,1))
y_train = np.float32(mat['y_train'])
y_train = np.reshape(y_train,(n_train,out_siz,1))

n_train = 1
x_train = x_train[0:n_train,:,:]
y_train = y_train[0:n_train,:,:]

print('Data File Name: %s' % (data_fname))
print(np.shape(x_train))
print(np.shape(y_train))

#=========================================================
#----- Parameters Setup

prefixed = True

#----- Tunable Parameters of BNet
channel_siz = 32 # Num of interp pts on each dim

#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
nlvl = 2*math.floor(math.log(min(in_siz,out_siz),2)/2)
# Filter size for the input and output
in_filter_siz = int(in_siz/2**nlvl)
out_filter_siz = int(out_siz/2**nlvl)

print("======== Parameters =========")
print("Channel Size:  %6d" % (channel_siz))
print("Num Levels:    %6d" % (nlvl))
print("Prefix Coef:   %6r" % (prefixed))
print("In Range:     (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
print("Out Range:    (%6.2f, %6.2f)" % (out_range[0], out_range[1]))

#=========================================================
#----- Variable Preparation
sess = tf.Session()

trainInData = tf.placeholder(tf.float32, shape=(n_train,in_siz,1),
        name="trainInData")
trainOutData = tf.placeholder(tf.float32, shape=(n_train,out_siz,1),
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

#=========================================================
#----- Step by Step Training
sess.run(init)
train_dict = {trainInData: x_train, trainOutData: y_train}
train_loss = sess.run(loss_train,feed_dict=train_dict)
print("Train Loss: %10e." % (train_loss))
sess.close()