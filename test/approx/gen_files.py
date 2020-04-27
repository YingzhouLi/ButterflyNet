import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import numpy as np

freq_list     = [64,256]
Lk_list       = range(1,4)

workdir = os.path.abspath(os.getcwd())

script_folder = './approx/'

shfolder   = script_folder+'/script/sh/'
os.makedirs(shfolder,   exist_ok=True)

with open(script_folder+'/template.json') as json_file:
    paras = json.load(json_file)

allpath = shfolder+'/all.sh'
with open(allpath,'w') as sh_file:
    sh_file.write('#!/bin/bash\n')
os.chmod(allpath, 0o775)

for freq_max in freq_list:
    Lmax = int(np.log2(freq_max))
    L_list = range(Lmax-2, Lmax+1)
    freq = [0, freq_max]
    for L in L_list:
        for Lk in Lk_list:
            freqstr = 'freq'+str(freq_max)
            Lstr    = 'L'+str(L)
            Lkstr   = 'Lk'+str(Lk)

            namefolder = script_folder \
                    + '/' + freqstr \
                    + '/' + Lstr \
                    + '/' + Lkstr \
            namestr = freqstr \
                    + '_' + Lstr \
                    + '_' + Lkstr \

            paras['neural network']['num of layers before switch'] \
                    = L-Lk
            paras['neural network']['num of layers after switch'] \
                    = Lk
            paras['neural network']['output range'] \
                    = freq

            jsonpath = namefolder + '/para.json'
            os.makedirs(os.path.dirname(jsonpath), exist_ok=True)
            with open(jsonpath,'w') as json_file:
                json.dump(paras,json_file,indent=4)

            shpath = shfolder+'/'+namestr+'.sh'
            with open(shpath,'w') as sh_file:
                sh_file.write('#!/bin/bash\n')
                sh_file.write('cd %s\n'%workdir)
                sh_file.write('python3 -u run_approx.py ')
                sh_file.write('%s '%jsonpath)
                sh_file.write('| tee %s/output.out'%namefolder)
            os.chmod(shpath, 0o775)

            with open(allpath,'w+') as sh_file:
                sh_file.write('./%s\n', %(namestr+'.sh'))
