import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import json
import numpy as np

nn_type_list  = ['bnet', 'ibnet']
Lk_list       = range(1,4)
prefixed_list = [True, False]
freq_list     = [[0,128],[256,384]]

with open('template.json') as json_file:
    paras = json.load(json_file)

for nn_type in nn_type_list:
    for Lk in Lk_list:
        for prefixed in prefixed_list:
            for freqit in range(len(freq_list)):
                freq = freq_list[freqit]
                paras['neural network']['neural network type'] = nn_type
                paras['neural network']['prefixed'] = prefixed
                paras['neural network']['num of layers before switch'] \
                        = 9-Lk
                paras['neural network']['num of layers after switch'] = Lk
                paras['neural network']['output range'] = freq

                if prefixed:
                    prefixstr = 'prefix'
                else:
                    prefixstr = 'random'

                if freqit == 0:
                    freqstr = 'lfreq'
                else:
                    freqstr = 'hfreq'

                namestr = nn_type+'_'+freqstr+'_Lk' \
                        +str(Lk)+'_'+prefixstr

                paras['train and test']['save model path'] = \
                        './dft/saved_models/'+namestr
                with open('json/'+namestr+'.json','w') as json_file:
                    json.dump(paras,json_file,indent=4)

                workdir = os.path.abspath(
                        os.path.join(os.getcwd(),os.pardir))

                with open('script/'+namestr+'.sh','w') as sh_file:
                    sh_file.write('#!/bin/bash\n')
                    sh_file.write('cd %s\n'%workdir)
                    sh_file.write('python3 -u run_dft.py dft/json/%s.json '%namestr)
                    sh_file.write('| tee dft/output/%s.out'%namestr)
                os.chmod('script/'+namestr+'.sh', 0o775)

                with open('script/'+namestr+'_qsub.sh','w') as sh_file:
                    sh_file.write('#!/bin/bash\n')
                    sh_file.write('#$ -N %s\n'%namestr)
                    sh_file.write('#$ -wd %s\n'%workdir)
                    sh_file.write('#$ -o %s\n'%(workdir+'/dft/output'))
                    sh_file.write('#$ -e %s\n\n'%(workdir+'/dft/error'))
                    sh_file.write('python3 -u run_dft.py dft/json/%s.json'%namestr)

