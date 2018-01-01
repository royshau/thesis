# !/usr/bin/python
import os
import numpy as np


def print_results_from_matlab(res_path, name):
    """

    :param res_path:
    :param name:
    :return:
    """
    # Read all data
    with open(res_path, 'r') as f:
        data = f.readlines()

    data = [dat.split("=") for dat in data]
    data_zero = np.array([float(dat[1]) for dat in data if "Zero" in dat[0]])
    data_name = np.array([float(dat[1]) for dat in data if name in dat[0]])

    print("PSNR-MEAN-ZERO = %f [dB]" % (data_zero.mean()))
    print("PSNR-MEAN-%s= %f [dB]" % (name, data_name.mean()))

    print("PSNR-STD-ZERO = %f [dB]" % (data_zero.std()))
    print("PSNR-STD-%s = %f [dB]" % (name, data_name.std()))


if __name__ == '__main__':
    res_path = '/media/ohadsh/sheard/googleDrive/Master/runs/results_text/cs_reported_miccai.txt'
    name = 'CS'
    print_results_from_matlab(res_path=res_path, name=name)