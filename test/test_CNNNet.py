import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
import math
import numpy as np
import scipy.io as spio
import tensorflow as tf
from matplotlib import pyplot as plt

from CNNLayer import CNNLayer


#=========================================================
# Read data from file
#---------------------------------------------------------
data_fname = 'data_DFT_smooth_025600.mat'

mat = spio.loadmat(data_fname,squeeze_me=False)

n_train = int(mat['n_train'])
n_test = int(mat['n_test'])
in_siz = int(mat['in_siz']) # Length of input vector
out_siz = int(mat['out_siz']) # Length of output vector
in_range = np.squeeze(np.float32(mat['in_range']))
out_range = np.squeeze(np.float32(mat['out_range']))
x_train = np.float32(mat['x_train'])
x_train = np.reshape(x_train,(n_train,in_siz,1))
y_train = np.float32(mat['y_train'])
y_train = np.reshape(y_train,(n_train,out_siz,1))
x_test = np.float32(mat['x_test'])
x_test = np.reshape(x_test,(n_test,in_siz,1))
y_test = np.float32(mat['y_test'])
y_test = np.reshape(y_test,(n_test,out_siz,1))

print("========== Inputs ===========")
print('Data File Name: %s' % (data_fname))
print("X train shape:   (%6d, %6d)" %
        (x_train.shape[0],x_train.shape[1]) )
print("Y train shape:   (%6d, %6d)" %
        (y_train.shape[0],y_train.shape[1]) )
print("X test shape:    (%6d, %6d)" %
        (x_test.shape[0],x_test.shape[1]) )
print("Y test shape:    (%6d, %6d)" %
        (y_test.shape[0],y_test.shape[1]) )

#=========================================================
#----- Parameters Setup

prefixed = True

#----- Tunable Parameters of BNet
batch_siz = 100 # Batch size during traning
channel_siz = 8 # Num of interp pts on each dim

adam_learning_rate = 0.01
adam_beta1 = 0.9
adam_beta2 = 0.999

max_iter = 500000 # Maximum num of iterations
report_freq = 1 # Frequency of reporting

#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
nlvl = 2*math.floor(math.log(min(in_siz,out_siz//2),2)/2)
# Filter size for the input and output
in_filter_siz = in_siz//2**nlvl
out_filter_siz = out_siz//2**nlvl

print("======== Parameters =========")
print("Batch Size:       %6d" % (batch_siz))
print("Channel Size:     %6d" % (channel_siz))
print("ADAM LR:          %6.4f" % (adam_learning_rate))
print("ADAM Beta1:       %6.4f" % (adam_beta1))
print("ADAM Beta2:       %6.4f" % (adam_beta2))
print("Max Iter:         %6d" % (max_iter))
print("Num Levels:       %6d" % (nlvl))
print("In Range:        (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
print("Out Range:       (%6.2f, %6.2f)" % (out_range[0], out_range[1]))

#=========================================================
#----- Variable Preparation
sess = tf.Session()

trainInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
        name="trainInData")
trainOutData = tf.placeholder(tf.float32, shape=(batch_siz,out_siz,1),
        name="trainOutData")
testInData = tf.placeholder(tf.float32, shape=(n_test,in_siz,1),
        name="testInData")
testOutData = tf.placeholder(tf.float32, shape=(n_test,out_siz,1),
        name="testOutData")

#=========================================================
#----- Training Preparation
cnn_net = CNNLayer(in_siz, out_siz,
        in_filter_siz, out_filter_siz,
        channel_siz, nlvl, prefixed )
y_train_output = cnn_net(tf.convert_to_tensor(x_train))
y_test_output = cnn_net(tf.convert_to_tensor(x_test))

loss_train = tf.reduce_mean(
        tf.squared_difference(y_train, y_train_output))
loss_test = tf.reduce_mean(
        tf.squared_difference(y_test, y_test_output))

optimizer_adam = tf.train.AdamOptimizer(adam_learning_rate,
        adam_beta1, adam_beta2)
train_step = optimizer_adam.minimize(loss_train)

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training
sess.run(init)

hist_loss_train = []
hist_loss_test = []
for it in range(max_iter):
    rand_index = np.random.choice(n_train, size=batch_siz)
    rand_x = x_train[rand_index,:,:]
    rand_y = y_train[rand_index,:,:]
    train_dict = {trainInData: rand_x, trainOutData: rand_y}
    if it % report_freq == 0:
        temp_train_loss = sess.run(loss_train,feed_dict=train_dict)
        test_dict = {testInData: x_test, testOutData: y_test}
        temp_test_loss = sess.run(loss_test,feed_dict=test_dict)
        hist_loss_train.append(temp_train_loss)
        hist_loss_test.append(temp_test_loss)
        print("Iter # %6d: Train Loss: %10e; Test Loss: %10e." % (it+1,
                temp_train_loss, temp_test_loss))
        sys.stdout.flush()
    sess.run(train_step, feed_dict=train_dict)

sess.close()
