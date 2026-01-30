#!/usr/bin/python3
"""
submit batch jobs to the queue
"""

from subprocess import check_output
from functools import reduce
from itertools import product, repeat
from copy import deepcopy
from time import sleep
from os import path,mkdir
from sys import argv,exit


OUTPUT_DIR = "/n/netscratch/yao_lab/Everyone/Francisco/QuantumNetworks/MER/"
folder = input('output subdirectory name: ')
OUTPUT_DIR = path.join(OUTPUT_DIR, folder)

if '-dry-run' not in argv:
    if not path.isdir(OUTPUT_DIR):
        mkdir(OUTPUT_DIR)
    else:
        if input('directory exists. overwrite? ') not in ['y','Y']:
            exit()

def get_lists(keys_so_far, d):
    sub_lists = []
    for key, val in d.items():
        if isinstance(val, list):
            sub_lists.append( [keys_so_far+[key],val] )
        elif isinstance(val, dict):
            sub_lists += get_lists(keys_so_far+[key],val)
        else:
            sub_lists.append( [keys_so_far+[key],[val]] )
    return sub_lists

def make_product( d ):
    '''
    Take a dictionary and return the product of options in the lists.
    I know that didn't make much sense. it does in my head.
    '''
    lsts = get_lists([],d)

    return product(*[zip(repeat(x[0]),x[1]) for x in lsts])

def get_nested_dict( d, key_list ):
    return reduce(dict.__getitem__,key_list,d)


options = {
    'batch_options' : {
        '-J':'GenerateGraphs',
        '-p':'serial_requeue',
        '-n':'1',
        '-N':'1',
        '-t':'0-16:00:00',
        '--mem-per-cpu':'1G',
        '--account':'yao_lab',
        '--contiguous':'',
        '--open-mode=append': '',
        '--requeue':'',
    },

    'run_options' : {
        "rep" : [str(i) for i in range(1730)]
    }
}

# output these options so that we remember what they were
if '-dry-run' not in argv:
    with open(path.join(OUTPUT_DIR,'run_options'),'w') as f:
        f.write('batch_options:\n')
        for opt,val in options['batch_options'].items():
            f.write('   '+str(opt)+':'+str(val)+'\n')
        f.write('run_options:\n')
        for opt,val in options['run_options'].items():
            f.write('   '+str(opt)+':'+str(val)+'\n')

run_lst = []
for n,x in enumerate(make_product(options)):
    tmp_d = {'batch_options':{},'run_options':{}}
    for key_list,value in x:
        if isinstance(value,dict): # don't clobber other stuff in a dict there
            get_nested_dict(tmp_d,key_list[:-1])[key_list[-1]].update(value)
        else: # this is just a regular value
            get_nested_dict(tmp_d,key_list[:-1])[key_list[-1]] = value

    # give the job a unique name and places to output stuff
    tmp_d['batch_options']['-o'] = path.join(OUTPUT_DIR, '%a_'+tmp_d['batch_options']['-J']+'.out')
    tmp_d['batch_options']['-e'] = path.join(OUTPUT_DIR, '%a_'+tmp_d['batch_options']['-J']+'.err')
    tmp_d['batch_options']['-J'] = OUTPUT_DIR.split('/')[-1]+'_'+tmp_d['batch_options']['-J']

    #tmp_d["run_options"]["rep"] = str(n)
    run_lst.append(tmp_d)


for job_n,d in enumerate(run_lst):

    s = ''
    for opt,val in d['run_options'].items():
        s += ','.join([opt,val])+'\n'

    if not '-dry-run' in argv:
        with open(path.join(OUTPUT_DIR,str(job_n))+'.opts','w') as f:
            f.write(s)
    else:
        print(job_n,'\n  ',end='')
        print(s)
        print()

batch_script = '#!/bin/bash'

for opt,val in d['batch_options'].items():
    batch_script += ' '.join(['\n#SBATCH',opt,val])

batch_script += "\n\n"
batch_script += 'echo \"Job Id: $SLURM_JOB_ID\"'
batch_script += "\n"
batch_script += 'echo \"Array Job Id: $SLURM_ARRAY_JOB_ID\"'
batch_script += "\n"
batch_script += 'echo \"Array Taks Id: $SLURM_ARRAY_TASK_ID\"'
batch_script += "\n"
batch_script += "\n"
batch_script += "\n"
batch_script += "source /n/home03/fmachado/QuantumNetworks/DelayedDecoder_Cluster/venv/bin/activate\n"
batch_script += "python /n/home03/fmachado/QuantumNetworks/DelayedDecoder_Cluster/RunningDecoderLaptop.py 16_1_6_b "
batch_script += ' $((SLURM_ARRAY_TASK_ID * 32)) $((SLURM_ARRAY_TASK_ID * 32 + 32))\n\n'

if '-dry-run' in argv:
    print(batch_script)
else:
    # write out the batch script to keep track of what we did
    with open(path.join(OUTPUT_DIR,d['batch_options']['-J'])+'.batch','w') as f:
        f.write(batch_script)
        
    print(check_output(['sbatch','--array=0-'+str(len(run_lst)-1)],
                       input=batch_script,
                       universal_newlines=True),
          '('+str(len(run_lst))+')')

