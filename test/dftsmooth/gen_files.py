import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import numpy as np

nn_type_list  = ['bnet', 'ibnet']
c_siz_list    = [16]
Lk_list       = range(1,4)
prefixed_list = [True, False]
freq_list     = [[0,128],[256,384]]

workdir = os.path.abspath(os.getcwd())

script_folder = './dftsmooth/'

outfolder  = script_folder+'/script/out/'
errfolder  = script_folder+'/script/err/'
shfolder   = script_folder+'/script/sh/'
qsubfolder = script_folder+'/script/qsub/'
os.makedirs(outfolder,  exist_ok=True)
os.makedirs(errfolder,  exist_ok=True)
os.makedirs(shfolder,   exist_ok=True)
os.makedirs(qsubfolder, exist_ok=True)

with open(script_folder+'/template.json') as json_file:
    paras = json.load(json_file)

for nn_type in nn_type_list:
    for c_siz in c_siz_list:
        for Lk in Lk_list:
            for prefixed in prefixed_list:
                for freqit in range(len(freq_list)):
                    freq = freq_list[freqit]

                    if prefixed:
                        prefixstr = 'prefix'
                    else:
                        prefixstr = 'random'

                    if freqit == 0:
                        freqstr = 'low_freq'
                    else:
                        freqstr = 'high_freq'

                    namefolder = script_folder \
                            + '/' + freqstr \
                            + '/' + nn_type \
                            + '/channel_size'+str(c_siz) \
                            +'/Lk'+ str(Lk) \
                            + '/' + prefixstr
                    namestr = freqstr \
                            + '_' + nn_type \
                            + '_csiz'+str(c_siz) \
                            +'_Lk'+ str(Lk) \
                            + '_' + prefixstr

                    paras['neural network']['neural network type'] \
                            = nn_type
                    paras['neural network']['channel size'] \
                            = c_siz
                    paras['neural network']['prefixed'] \
                            = prefixed
                    paras['neural network']['num of layers before switch'] \
                            = 8-Lk
                    paras['neural network']['num of layers after switch'] \
                            = Lk
                    paras['neural network']['output range'] \
                            = freq
                    paras['train and test']['save folder path'] \
                            = namefolder
                    if prefixed:
                        paras['train and test']['max num of iteration'] \
                                = 20000
                        paras['train and test']['exponential decay'] \
                                ['initial learning rate'] \
                                = 2e-5
                    else:
                        paras['train and test']['max num of iteration'] \
                                = 50000
                        paras['train and test']['exponential decay'] \
                                ['initial learning rate'] \
                                = 1e-3


                    jsonpath = namefolder + '/para.json'
                    os.makedirs(os.path.dirname(jsonpath), exist_ok=True)
                    with open(jsonpath,'w') as json_file:
                        json.dump(paras,json_file,indent=4)


                    shpath = shfolder+'/'+namestr+'.sh'
                    with open(shpath,'w') as sh_file:
                        sh_file.write('#!/bin/bash\n')
                        sh_file.write('cd %s\n'%workdir)
                        sh_file.write('python3 -u run_dft.py ')
                        sh_file.write('%s '%jsonpath)
                        sh_file.write('| tee %s/output.out'%namefolder)
                    os.chmod(shpath, 0o775)

                    qsubpath = qsubfolder+'/'+namestr+'.sh'
                    with open(qsubpath,'w') as sh_file:
                        sh_file.write('#!/bin/bash\n')
                        sh_file.write('#$ -N %s\n'%namestr)
                        sh_file.write('#$ -wd %s\n'%workdir)
                        sh_file.write('#$ -o %s\n'%outfolder)
                        sh_file.write('#$ -e %s\n\n'%errfolder)
                        sh_file.write('python3 -u run_dft.py ')
                        sh_file.write('%s '%jsonpath)
                        sh_file.write('| tee %s/output.out'%namefolder)
