"""
MRI data base - all needed for handle my MRI database, NIfTI and DICOM as well
"""
import json
import os
from collections import OrderedDict
import glob
import nibabel as nib
import numpy as np

from common.viewers.orthoslicer import OrthoSlicer3D

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_to_path.json'), 'r') as f:
    data_to_path = json.load(f)


class MriDataBase:
    """
    Mri data base class - only source data, for nifty or dicom
    """

    def __init__(self, data_name, nifti_or_dicom='nifti'):
        """
        Constructor
        :param data_name: data name
        :param nifti_or_dicom: nifty / dicom
        :return: MriDataBase object
        """
        self.data_name = data_name
        self.nifty_or_dicom = nifti_or_dicom
        self.data_path = data_to_path[data_name]["data"]
        self.labels_path = data_to_path[data_name]["labels"]
        # New method for ADNI
        if "suffix" in data_to_path[data_name]:
            self.items = sorted(glob.glob(os.path.join(data_to_path[data_name]["data"], data_to_path[data_name]["suffix"])))
            self.items = ['/'.join(item.split('/')[-2:]) for item in self.items]
            # self.items = [item.strip(self.data_path) for item in self.items]
        else:
            # OLD SchizReg
            file_suffix = "hdr" if nifti_or_dicom == 'nifti' else 'dicom'  # TODO
            self.items = sorted([item for item in os.listdir(data_to_path[data_name]["data"]) if file_suffix in item])

        self.data = dict(img=[],meta_data=[], labels=[])
        self.info = self.set_info(data_to_path[data_name]["info"])

    def get_source_data(self, item='all'):
        """
        Read item from data path. If item=='all', read all items
        :param item: item name or all
        :return: dictionary
        """
        items = self.items if item=='all' else [item]
        data = dict(img=[], meta_data=[], labels=[])

        for it in items:

            if self.nifty_or_dicom == 'nifti':
                niftii_obj = nib.load(os.path.join(self.data_path, it))
                data["meta_data"].append(OrderedDict(niftii_obj.header))
                data["labels"] = 'labels'
                dat = niftii_obj.get_data()
                data['img'].append(dat)
        return data

    def set_info(self, info_path):
        """
        Set info for dataBase.
        :param info_path: info path
        :return: dictionary
        """
        ret_dic = {}
        try:
            # NEW METHOD
            with open(os.path.join(info_path, 'data_info.json'), 'r') as f:
                info = json.load(f)
            ret_dic["case_to_hash"] = {case: meta['hash'] for (case,meta) in info.iteritems()}
            ret_dic["hash_to_case"] = {meta['hash']: case for (case,meta) in info.iteritems()}
            ret_dic["train_test_list"] = {case: meta['tt'] for (case, meta) in info.iteritems()}
            ret_dic["file"] = {case: meta['file'] for (case, meta) in info.iteritems()}
        except:
            # OLD METHOD
            with open(os.path.join(info_path, 'case_to_hash.json'), 'r') as f:
                ret_dic["case_to_hash"] = json.load(f)

            with open(os.path.join(info_path, 'train_test_list.json'), 'r') as f:
                ret_dic["train_test_list"] = json.load(f)

        return ret_dic

    def show_data(self, data):
        """
        Show data
        :param data : 3d matrix
        :return:
        """
        OrthoSlicer3D(data)

