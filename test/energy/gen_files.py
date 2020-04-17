import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import numpy as np

nn_type_list  = ['bnet', 'ibnet']
c_siz_list    = [16]
Lk_list       = range(1,4)
init_list     = ['dft', 'glorot_uniform']
task_list     = ['squaresum', 'singlefullyconnect']

workdir = os.path.abspath(os.getcwd())

script_folder = './energy/'

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
            for init in init_list:
                for task in task_list:

                    if init == 'dft':
                        prefixstr = 'prefix'
                    else:
                        prefixstr = 'random'

                    if task == 'squaresum':
                        taskstr = 'sqr'
                    else:
                        taskstr = 'den'

                    namefolder = script_folder \
                            + '/' + taskstr \
                            + '/' + nn_type \
                            + '/channel_size'+str(c_siz) \
                            +'/Lk'+ str(Lk) \
                            + '/' + prefixstr
                    namestr = taskstr \
                            + '_' + nn_type \
                            + '_csiz'+str(c_siz) \
                            +'_Lk'+ str(Lk) \
                            + '_' + prefixstr

                    paras['neural network']['neural network type'] \
                            = nn_type
                    paras['neural network']['task layer type'] \
                            = task
                    paras['neural network']['channel size'] \
                            = c_siz
                    paras['neural network']['initializer'] \
                            = init
                    paras['neural network']['num of layers before switch'] \
                            = 8-Lk
                    paras['neural network']['num of layers after switch'] \
                            = Lk
                    paras['train and test']['save folder path'] \
                            = namefolder
                    if init == 'dft':
                        paras['train and test']['max num of iteration'] \
                                = 50000
                        if taskstr == 'sqr':
                            paras['train and test']['exponential decay'] \
                                    ['initial learning rate'] \
                                    = 1e-4
                        else:
                            paras['train and test']['exponential decay'] \
                                    ['initial learning rate'] \
                                    = 1e-3
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
                        sh_file.write('python3 -u run_energy.py ')
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
                        sh_file.write('python3 -u run_energy.py ')
                        sh_file.write('%s '%jsonpath)
                        sh_file.write('| tee %s/output.out'%namefolder)
