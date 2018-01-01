#!/usr/bin/python
import os
import argparse
import nibabel as nib
import numpy as np
REFERECE = '/media/ohadsh/sheard/Ohad/thesis/tools/examples/IXI389-Guys-0930-T1.nii.gz'


def write_nifti_data(data, output_path, reference=REFERECE, name=None):
    """
    Write binary file to nifti
    :param data:
    :param output_path:
    :param reference:
    :param name:
    :return:
    """

    nii_ref = nib.load(reference)
    new_nii = nib.Nifti1Image(data, nii_ref.affine)
    new_nii.update_header()
    name = 'new' if name is None else name
    nib.save(new_nii, os.path.join(output_path, name+'.nii.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nifti write.')
    parser.add_argument('--data', dest='data', default='', type=np.array, help='binary data')
    parser.add_argument('--output_path', dest='output_path', type=str, help='nifti output path')
    parser.add_argument('--reference', dest='reference', type=str, help='nifti reference path')
    parser.add_argument('--name', dest='name', type=str, help='output name')
    args = parser.parse_args()
    write_nifti_data(os.path.abspath(args.data_path), os.path.abspath(args.output_path))

