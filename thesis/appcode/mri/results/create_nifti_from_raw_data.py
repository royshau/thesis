# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.data.write_nifti_data import write_nifti_data
from appcode.mri.data.mri_data_base import MriDataBase
from common.files_IO.file_handler import FileHandler
from appcode.mri.k_space.utils import get_image_from_kspace
from appcode.mri.k_space.data_creator import get_rv_mask

file_names = ['k_space_real_gt', 'k_space_imag_gt', 'meta_data']
import argparse

predict_info = {'width': 256, 'height': 256, 'channels': 1, 'dtype': 'float32'}
predict_names = {'real': '000000.predict_real.bin', 'imag': '000000.predict_imag.bin'}
import matplotlib.pyplot as plt

META_KEYS = {'hash':0, 'slice': 1, 'bit_pix':2, 'aug':3, 'norm_factor':4}
MASKS_DIR = '/media/ohadsh/Data/ohadsh/work/matlab/thesis/'


def create_nifti_from_raw_data(data_dir, predict_path, output_path, data_base, batch_size, num_of_cases=-1,
                               tt='train', source='k_space', random_sampling_factor=None, cs_path=None):
    """
    Assumption - predict on all examples exists
    This script create nifti files from k-space raw data, original and predictions.
    :param data_dir:
    :param predict_path:
    :param output_path:
    :param data_base:
    :param batch_size:
    :param num_of_cases:
    :param tt:
    :param random_sampling_factor:
    :param cs_path: compressed sensing predicted path
    :return:
    """

    db = MriDataBase(data_base)

    f_predict = {}
    cs_pred = None
    for name_pred in ['real', 'imag']:
        f_predict[name_pred] = FileHandler(path=os.path.join(predict_path, predict_names[name_pred]),
                                                   info=predict_info, read_or_write='read', name=name_pred, memmap=True)
    if cs_path is not None:
        cs_pred = FileHandler(path=cs_path, info=predict_info, read_or_write='read', name='CS', memmap=True)
        # write_nifti_data(cs_pred.memmap.transpose(2, 1, 0), output_path='/tmp/', name='CS')

    data_set = KspaceDataSet(data_dir, file_names, stack_size=batch_size, shuffle=False, data_base=data_base, memmap=True)

    data_set_tt = getattr(data_set, tt)

    meta_data = data_set_tt.files_obj['meta_data'].memmap

    # Get all unique case hash
    all_cases = np.unique(meta_data[:, META_KEYS['hash']])
    all_cases = all_cases if num_of_cases == -1 else all_cases[:num_of_cases]

    # For each case, create indices, build a nifty from real image and predict
    done = 1
    for case in all_cases:
        try:
            idx = get_case_idx(case, meta_data)
            name = db.info['hash_to_case'][case]
            print("Working on case : %s, number= (%d / %d)" % (name, done, num_of_cases))
            ref = os.path.join(db.data_path, name, "IXI"+name+".nii.gz")

            if not os.path.exists(ref):
                ref = None

            res_out_path = os.path.join(output_path, name)
            if not os.path.exists(res_out_path):
                os.makedirs(res_out_path)

            # Data creation
            org_real = data_set_tt.files_obj['k_space_real_gt'].memmap[idx]
            org_imag = data_set_tt.files_obj['k_space_imag_gt'].memmap[idx]
            data = get_image_from_kspace(org_real, org_imag).transpose(1, 2, 0)
            # data = norm_data(data)
            write_nifti_data(data, output_path=res_out_path, reference=ref, name=name)

            # Predict from network
            pred_real = f_predict['real'].memmap[idx]
            pred_imag = f_predict['imag'].memmap[idx]

            if source == 'k_space':
                data = get_image_from_kspace(pred_real, pred_imag).transpose(2, 1, 0)
            else:
                data = 256*np.abs(pred_real+ 1j * pred_imag).transpose(2, 1, 0)
            # data = norm_data(data)
            write_nifti_data(data, output_path=res_out_path, reference=ref, name=name+"_predict")

            # Zero Padding
            if random_sampling_factor is not None:
                mask = get_rv_mask(mask_main_dir=MASKS_DIR, factor=random_sampling_factor)
                org_real_zero_padded = mask * org_real
                org_imag_zero_padded = mask * org_imag
                data = get_image_from_kspace(org_real_zero_padded, org_imag_zero_padded).transpose(1, 2, 0)
                # data = norm_data(data)
                write_nifti_data(data, output_path=res_out_path, reference=ref, name=name+"_zeroPadding")

            # CS
            if cs_pred is not None:
                data = cs_pred.memmap[idx].transpose(2, 1, 0)
                # data = norm_data(data)
                write_nifti_data(data, output_path=res_out_path, reference=ref, name=name + "_CS")

            done += 1
        except:
            print "BAD: (min, max) = (%d, %d)" % (idx.min(), idx.max())
            continue


def get_case_idx(case_hash, meta_data):
    """ Get case indices given cash hash and meta data memmap
    :param case_hash:
    :param meta_data:
    :return:
    """
    idx = np.where(meta_data[:, META_KEYS['hash']] == case_hash)[0]
    slice_idx_rel = np.argsort(meta_data[idx, META_KEYS['slice']])
    slice_idx_abs = idx[slice_idx_rel]
    return slice_idx_abs


def norm_data(data):
    """
    Normalize data
    :param data:
    :return:
    """
    norm_factor = 1.0 / data.max()
    return (data * norm_factor).astype('float32')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='TBD.')
    parser.add_argument('--tt', dest='tt', choices=['train', 'test'], default='train', type=str, help='train / test')
    parser.add_argument('--data_dir', dest='data_dir', default='/media/ohadsh/Data/ohadsh/work/data/T1/sagittal/', type=str, help='data directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=50,  help='mini batch size')
    parser.add_argument('--data_base', dest='data_base', type=str, default='IXI_T1', help='data base name - for file info')
    parser.add_argument('--predict_path', dest='predict_path', type=str, help='run path')
    parser.add_argument('--output_path', dest='output_path', default='./', type=str, help='out path')
    parser.add_argument('--source', dest='source', default='k_space', type=str, help='source')
    parser.add_argument('--random_sampling_factor', dest='random_sampling_factor', type=int, default=None,
                        help='Random sampling factor for zero padding')
    parser.add_argument('--cs_path', dest='cs_path', default=None, type=str, help='CS path')
    args = parser.parse_args()

    create_nifti_from_raw_data(args.data_dir, args.predict_path, args.output_path,
         args.data_base, args.batch_size, args.num_of_cases, args.tt, args.source,
                               args.random_sampling_factor, args.cs_path)