import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
from pathlib import Path
import numpy as np
import scipy.io as spio

strnamelist = [
        "test_BNet_P1_ys",   "test_BNet_P1_ys_true",
        "test_CNNNet_P1_ys", "test_CNNNet_P1_ys_true",
        "train_BNet_P1_ys",   "train_BNet_P1_ys_true",
        "train_CNNNet_P1_ys", "train_CNNNet_P1_ys_true",
        ]
for strname in strnamelist:
    filehandle = Path("./tftmp/"+strname+".npy")
    data = np.load(filehandle)
    spio.savemat("./tftmp/"+strname+".mat", {strname: data})
