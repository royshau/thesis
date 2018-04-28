import os
import sys
import numpy.random
import numpy as np
import copy
import json
import shutil

with open('t1.txt', 'r') as f:
	ref = f.readlines()

for line in ref:
	line = line.split('\n')[0]
	dirname = line.split('.')[0].split('*')
	dirname = dirname[0]
	dirname = dirname[3:]
	path_start = os.getcwd()
	print(dirname)
	src = "%s" %line
	dst = os.path.join(path_start,"%s" % dirname,line)
	src = os.path.join(path_start,src)
	print(src)
	print(dst)
	os.mkdir(dirname)
	shutil.copy2(src,dst)
