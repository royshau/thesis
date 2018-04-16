import os
import sys
import numpy.random
import numpy as np
import copy
import json
import shutil

with open('../t1.txt', 'r') as f:
	ref = f.readlines()

for line in ref:
	line = line.split('\n')[0]
	dirname = line.split('.')[0].split('*')
	os.mkdir(dirname)
	shutil.copy("./%s" % line, os.path.join("./%s" % dirname, line))