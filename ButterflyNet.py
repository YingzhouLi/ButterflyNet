import math
import numpy as np 
import scipy.io as spio
import tensorflow as tf
from matplotlib import pyplot as plt

#=========================================================
# Read data from file
#---------------------------------------------------------

mat = spio.loadmat('data_DFT.mat',squeeze_me=False)

n_train = int(mat['n_train'])
n_test = int(mat['n_test'])
in_siz = int(mat['in_siz']) # Length of input vector
out_siz = int(mat['out_siz']) # Length of output vector
x_train = np.float32(mat['x_train'])
x_train = np.reshape(x_train,(n_train,in_siz,1))
y_train = np.float32(mat['y_train'])
y_train = np.reshape(y_train,(n_train,out_siz,1))
x_test = np.float32(mat['x_test'])
x_test = np.reshape(x_test,(n_test,in_siz,1))
y_test = np.float32(mat['y_test'])
y_test = np.reshape(y_test,(n_test,out_siz,1))
print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))

#=========================================================
#----- Parameters Setup

#----- Tunable Parameters of BNet
batch_siz = 10 # Batch size during traning
channel_siz = 12 # Num of interp pts on each dim

adam_learning_rate = 0.01
adam_beta1 = 0.9
adam_beta2 = 0.999

max_iter = 100000 # Maximum num of iterations
report_freq = 10 # Frequency of reporting

#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
nlvl = 2*math.floor(math.log(min(in_siz,out_siz),2)/2)
# Filter size for the input and output
in_filter_siz = int(in_siz/2**nlvl)
out_filter_siz = int(out_siz/2**nlvl)

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

tfInFilterVar = tf.Variable( tf.random_normal(
    [in_filter_siz, 1, channel_siz]), name="Filter_In" )
tfInBiasVar = tf.Variable( tf.zeros([channel_siz]), name="Bias_In" )

tfFilterVars = []
tfBiasVars = []
for lvl in range(0,nlvl):
    tmpFilterVars = []
    tmpBiasVars = []
    for itk in range(0,2**(lvl+1)):
        varLabel = "LVL_%02d_%04d" % (lvl, itk)
        filterVar = tf.Variable(
                tf.random_normal([2,channel_siz,channel_siz]), 
                name="Filter_"+varLabel )
        biasVar = tf.Variable(tf.zeros([channel_siz]),
                name="Bias_"+varLabel )
        tmpFilterVars.append(filterVar)
        tmpBiasVars.append(biasVar)
    tfFilterVars.append(list(tmpFilterVars))
    tfBiasVars.append(list(tmpBiasVars))

tfMidDenseVars = []
tfMidBiasVars = []
for itk in range(0,2**(int(nlvl/2))):
    tmpMidDenseVars = []
    tmpMidBiasVars = []
    for itx in range(0,2**(int(nlvl/2))):
        varLabel = "LVL_Mid_%04d_%04d" % (itk,itx)
        denseVar = tf.Variable(
                tf.random_normal([channel_siz,channel_siz]), 
                name="Dense_"+varLabel )
        biasVar = tf.Variable(tf.zeros([channel_siz]),
                name="Bias_"+varLabel )
        tmpMidDenseVars.append(denseVar)
        tmpMidBiasVars.append(biasVar)
    tfMidDenseVars.append(list(tmpMidDenseVars))
    tfMidBiasVars.append(list(tmpMidBiasVars))

tfOutFilterVars = []
for itk in range(0,2**(lvl+1)):
    varLabel = "Out_%04d" % (itk)
    filterVar = tf.Variable( tf.random_normal(
        [1, channel_siz, out_filter_siz]), name="Filter_"+varLabel )
    tfOutFilterVars.append(filterVar)

#=========================================================
#----- Structure Preparation
def butterfly_net(in_data):
    # coef_filter of size filter_size*in_channels*out_channels
    InInterp = tf.nn.conv1d(in_data, tfInFilterVar, stride=in_filter_siz,
            padding='VALID')
    InInterp = tf.nn.relu(tf.nn.bias_add(InInterp, tfInBiasVar))

    tfVars = []
    for lvl in range(0,int(nlvl/2)):
        tmpVars = []
        if lvl > 0:
            for itk in range(0,2**(lvl+1)):
                Var = tf.nn.conv1d(tfVars[lvl-1][math.floor(itk/2)],
                    tfFilterVars[lvl][itk],
                    stride=2, padding='VALID')
                Var = tf.nn.relu(tf.nn.bias_add(Var,
                    tfBiasVars[lvl][itk]))
                tmpVars.append(Var)
            tfVars.append(list(tmpVars))
        else:
            for itk in range(0,2**(lvl+1)):
                Var = tf.nn.conv1d(InInterp,
                    tfFilterVars[lvl][itk],
                    stride=2, padding='VALID')
                Var = tf.nn.relu(tf.nn.bias_add(Var,
                    tfBiasVars[lvl][itk]))
                tmpVars.append(Var)
            tfVars.append(list(tmpVars))

    lvl = int(nlvl/2) - 1
    for itk in range(0,2**(int(nlvl/2))):
        tmpVars = np.reshape([],(np.size(in_data,0),0,channel_siz))
        for itx in range(0,2**(int(nlvl/2))):
            tmpVar = tfVars[lvl][itk][:,itx,:]
            tmpVar = tf.matmul(tmpVar,tfMidDenseVars[itk][itx])
            tmpVar = tf.nn.relu( tf.nn.bias_add(
                tmpVar, tfMidBiasVars[itk][itx] ) )
            tmpVar = tf.reshape(tmpVar,
                    (np.size(in_data,0),1,channel_siz))
            tmpVars = tf.concat([tmpVars, tmpVar], axis=1)
        tfVars[lvl][itk] = tmpVars

    for lvl in range(int(nlvl/2),nlvl):
        tmpVars = []
        for itk in range(0,2**(lvl+1)):
            Var = tf.nn.conv1d(tfVars[lvl-1][math.floor(itk/2)],
                tfFilterVars[lvl][itk],
                stride=2, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                tfBiasVars[lvl][itk]))
            tmpVars.append(Var)
        tfVars.append(list(tmpVars))

    # coef_filter of size filter_size*in_channels*out_channels
    lvl = nlvl-1
    OutInterp = np.reshape([],(np.size(in_data,0),1,0))
    for itk in range(0,2**(lvl+1)):
        Var = tf.nn.conv1d(tfVars[lvl][itk],
            tfOutFilterVars[itk],
            stride=1, padding='VALID')
        OutInterp = tf.concat([OutInterp, Var], axis=2)

    out_data = tf.reshape(OutInterp,shape=(np.size(in_data,0),out_siz,1))

    return(out_data)


#=========================================================
#----- Training Preparation
y_train_output = butterfly_net(x_train)
y_test_output = butterfly_net(x_test)


loss_train = tf.reduce_mean(
        tf.squared_difference(y_train, y_train_output))
loss_test = tf.reduce_mean(
        tf.squared_difference(y_test, y_test_output))

optimizer_adam = tf.train.AdamOptimizer(adam_learning_rate,
        adam_beta1, adam_beta2)
train_step = optimizer_adam.minimize(loss_train)

# Initialize Variables
init = tf.global_variables_initializer()

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
    sess.run(train_step, feed_dict=train_dict)
    if (it+1) % report_freq == 0:
        temp_train_loss = sess.run(loss_train,feed_dict=train_dict)
        test_dict = {testInData: x_test, testOutData: y_test}
        temp_test_loss = sess.run(loss_test,feed_dict=test_dict)
        hist_loss_train.append(temp_train_loss)
        hist_loss_test.append(temp_test_loss)
        print("Iter # %6d: Train Loss: %10.4f; Test Loss: %10.4f." % (it+1,
                temp_train_loss, temp_test_loss))
