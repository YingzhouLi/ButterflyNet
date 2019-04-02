import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../../src")
sys.path.insert(0,"../../src/data_gen")
from pathlib import Path
import numpy as np
import scipy.io as spio

Ntest = 100000

# ========= Testing ============
for alpha in np.arange(0,5.01,0.2):
    x_test_data_file = Path("./tftmp/x_test_data_"+str(round(alpha,2))+".npy")
    y_test_data_file = Path("./tftmp/y_test_data_"+str(round(alpha,2))+".npy")
    x_test_data = np.load(x_test_data_file)
    y_test_data = np.load(y_test_data_file)

    ys_bnet = np.load("./tftmp/test_BNet_Alpha_"+str(round(alpha,2))+".npy")
    ys_cnn  = np.load("./tftmp/test_CNNNet_Alpha_"+str(round(alpha,2))+".npy")

    y_test_data = np.squeeze(y_test_data)
    ys_bnet = np.squeeze(ys_bnet)
    ys_cnn = np.squeeze(ys_cnn)

    err_bnet = np.linalg.norm(ys_bnet-y_test_data,axis=1)
    err_cnn = np.linalg.norm(ys_cnn-y_test_data,axis=1)

    print("BNet: %f (%f); CNN: %f (%f)" % ( np.mean(err_bnet),
            np.std(err_bnet), np.mean(err_cnn), np.std(err_cnn)) )
