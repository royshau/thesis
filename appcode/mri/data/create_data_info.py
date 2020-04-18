import json
import glob
import os

DataDir = r'/media/rrtammyfs/Users/Ariel/DCE_MRI/'
NumTest = 20
train =0
test=0

files = sorted(glob.glob(DataDir+'/stroke/*[!e]/nifti/*_??1_reg.hdr'))
files_test = sorted(glob.glob(DataDir+'/stroke/R_V-stroke/nifti/*.hdr'))
files += files_test
data = {}

for hash, filename in enumerate(files):
    casename = os.path.basename(filename).split(".")[-2]

    data[casename] = {}
    data[casename]['hash'] = hash
    data[casename]['file'] = filename
    if '20130417' in casename:
        data[casename]['tt'] = 'test'
        test +=1
    else:
        data[casename]['tt'] = 'train'
        train +=1

files = sorted(glob.glob(DataDir+'/tumors/[!N_I]*/nifti/*_??1_reg.hdr'))
files_test = sorted(glob.glob(DataDir+'/tumors/N_I-SOL/nifti/*.hdr'))
files += files_test

for hash, filename in enumerate(files):
    casename = os.path.basename(filename).split(".")[-2]

    data[casename] = {}
    data[casename]['hash'] = hash
    data[casename]['file'] = filename
    if 'n_i' in casename:
        data[casename]['tt'] = 'test'
        test +=1
    else:
        data[casename]['tt'] = 'train'
        train +=1

with open('data_info.json', 'w') as outfile:
    json.dump(data, outfile)

print("data has {} train volumes and {} test volumes".format(train,test))