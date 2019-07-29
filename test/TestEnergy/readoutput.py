import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../../src")
sys.path.insert(0,"../../src/data_gen")
from pathlib import Path
import numpy as np
import scipy.io as spio
import json

json_file = open('paras.json')
paras = json.load(json_file)

Ntest = paras['Ntest']

# ========= Testing ============
x_test_data_file = Path("./tftmp/x_test_data.npy")
y_test_data_file = Path("./tftmp/y_test_data.npy")
x_test_data = np.load(x_test_data_file)
y_test_data = np.load(y_test_data_file)

ys_bnet = np.load("./tftmp/test_BNet_Energy_128.npy")
ys_cnn  = np.load("./tftmp/test_CNNNet_Energy_128.npy")

y_test_data = np.squeeze(y_test_data)
ys_bnet = np.squeeze(ys_bnet)
ys_cnn = np.squeeze(ys_cnn)

err_bnet = np.absolute(ys_bnet-y_test_data)/y_test_data
err_cnn = np.absolute(ys_cnn-y_test_data)/y_test_data

print("BNet: %f (%f); CNN: %f (%f)" % ( np.mean(err_bnet),
        np.std(err_bnet), np.mean(err_cnn), np.std(err_cnn)) )
