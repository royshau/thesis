# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
import argparse
import nibabel as nib
NII_SUFFIX = '.nii.gz'
SEG_SUFFIX = '_seg'
BRAIN_SUFFIX = '_brain'
CLASSES = [0,1,2,3]


def calc_masked_psnr(data_dir, num_of_cases=-1, suffixes=None, brain_only=True):
    """ Go over segmentation results and calc
    TPR, TNR, ACC, DICE score
    :param data_dir:
    :param num_of_cases:
    :param suffixes:
    :param brain_only: calculate on brain only
    :return:
    """
    if brain_only:
        seg_suffix_use = BRAIN_SUFFIX+SEG_SUFFIX
    else:
        seg_suffix_use = SEG_SUFFIX

    suffixes = eval(suffixes) if suffixes is not None else None
    num_of_cases = 1000000000 if num_of_cases == -1 else num_of_cases
    sub_dirs = os.listdir(data_dir)
    PSNR = {suffix: [] for suffix in suffixes}
    MSE = {suffix: [] for suffix in suffixes}
    for sub_dir in sub_dirs:
        if sub_dirs.index(sub_dir) > num_of_cases - 1:
            break
        if not os.path.isdir(os.path.join(data_dir, sub_dir)):
            continue
        path_sub_dir = os.path.join(data_dir, sub_dir)
        all_files_in_dir = os.listdir(path_sub_dir)
        suffix_to_nifti = {}
        suffix_to_nifti_imag = {}

        print("Working on - %s, number: (%d / %d)" % (sub_dir, len(MSE[MSE.keys()[0]]), num_of_cases))
        # Collect all relevant files for this case
        for file_type in all_files_in_dir:
            for suffix in suffixes:
                if file_type.endswith(sub_dir+suffix+seg_suffix_use+NII_SUFFIX):
                    suffix_to_nifti[suffix] = nib.load(os.path.join(path_sub_dir, file_type))
                    suffix_to_nifti_imag[suffix] = nib.load(os.path.join(path_sub_dir, file_type.replace('_seg', '')))

        if '' in suffix_to_nifti.keys():
            mse = calc_statistics(suffix_to_nifti, suffix_to_nifti_imag)
            print sub_dir
            for suffix in suffix_to_nifti.keys():
                MSE[suffix].append(mse[suffix])

    # Print results
    print('Results\n')
    for suffix in PSNR.keys():
        mse_array = np.array(MSE[suffix])
        psnr_mean = calc_psnr(mse_array).mean(axis=0)
        psnr_std = calc_psnr(mse_array).std(axis=0)
        print("Version - %s" % suffix)
        # Print total result
        print("PSNR  = %f, %f ||| %s , %s" % (psnr_mean.mean(), psnr_std.mean(), psnr_mean, psnr_std))


def calc_statistics(suffix_to_nifti, suffix_to_nifti_imag):
    """
    Get dictionary - suffix to nifti, return statistics
    :param suffix_to_nifti:
    :return:
    """
    ref_seg = suffix_to_nifti[''].get_data()
    reg_img = suffix_to_nifti_imag[''].get_data()
    mse = {}
    psnr = {}
    num_of_cls = len(CLASSES)
    for (ver_name, ver_nifti) in suffix_to_nifti_imag.iteritems():
        if ver_name == '':
            mse[ver_name] = num_of_cls*[0.0]
            psnr[ver_name] = num_of_cls*[0.0]
            continue
        data = ver_nifti.get_data()
        mse_temp = []

        for cls in CLASSES:
            mask = (ref_seg == cls).astype(np.int)
            mse_cls = np.mean(np.sum((mask * (data - reg_img))**2, axis=(0,1)))
            mse_temp.append(mse_cls)

        mse[ver_name] = mse_temp

    return mse


def calc_psnr(mse):
    max_val = 256
    psnr = 20*np.log10(max_val / np.sqrt(mse))
    return psnr


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Go over segmentation results and calc statistics.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='data directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--suffixes', dest='suffixes', default=None, type=str, help='suffixes')
    parser.add_argument('--brain_only', dest='brain_only', default=True, type=bool, help='brain only')
    args = parser.parse_args()

    calc_masked_psnr(args.data_dir, args.num_of_cases, args.suffixes, args.brain_only)