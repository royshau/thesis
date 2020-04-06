import json
import glob
import os

DataDir = r'/media/rrtammyfs/Users/Ariel/DCE_MRI/stroke'
NumTest = 20

files = sorted(glob.glob(DataDir+'/*[!e]/nifti/*_0?1_reg.hdr'))
files_test = sorted(glob.glob(DataDir+'/R_V-stroke/nifti/*.hdr'))
files += files_test
data = {}
train =0
test=0
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

with open('data_info.json', 'w') as outfile:
    json.dump(data, outfile)

print("data has {} train volumes and {} test volumes".format(train,test))