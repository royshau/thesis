#!/usr/bin/python
import os
import argparse
import nibabel as nib
from common.viewers.orthoslicer import OrthoSlicer3D
import numpy as np

def show_nifti_data(data_path, data_path2=None):
    nii = nib.load(data_path)
    print nii.header
    print nii.shape
    if data_path2 is not None:
        nii2 = nib.load(data_path2)
        print nii2.header
        print nii2.shape
        data_to_show = np.abs(nii.get_data().squeeze() - nii2.get_data().squeeze())
        OrthoSlicer3D(data_to_show)
    else:
        OrthoSlicer3D(nii.get_data().squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nifti data viewer.')
    parser.add_argument('--data_path', dest='data_path', default='', type=str, help='nifti data path')
    parser.add_argument('--data_path2', dest='data_path2', default=None, type=str, help='nifti data path2')
    args = parser.parse_args()
    show_nifti_data(os.path.abspath(args.data_path), args.data_path2)
