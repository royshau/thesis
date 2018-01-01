# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
import argparse
import nibabel as nib
from appcode.mri.results.HausdorffDistance import ModHausdorffDist, HausdorffDist
from scipy.spatial.distance import directed_hausdorff, pdist
from scipy.ndimage.morphology import distance_transform_edt
NII_SUFFIX = '.nii.gz'
SEG_SUFFIX = '_seg'
BRAIN_SUFFIX = '_brain'
CLASSES = [0]


def calc_brain_statistics(data_dir, num_of_cases=-1, suffixes=None, brain_only=True):
    """ Go over brain extraction results and calculate housdorff distance
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
    MHD = {suffix: [] for suffix in suffixes}
    FHD = {suffix: [] for suffix in suffixes}
    RHD = {suffix: [] for suffix in suffixes}
    for sub_dir in sub_dirs:
        if sub_dirs.index(sub_dir) > num_of_cases - 1:
            break
        if not os.path.isdir(os.path.join(data_dir, sub_dir)):
            continue
        path_sub_dir = os.path.join(data_dir, sub_dir)
        all_files_in_dir = os.listdir(path_sub_dir)
        suffix_to_nifti = {}

        print("Working on - %s, number: (%d / %d)" % (sub_dir, len(MHD[MHD.keys()[0]]), num_of_cases))
        # Collect all relevant files for this case
        for file_type in all_files_in_dir:
            for suffix in suffixes:
                if file_type.endswith(sub_dir+suffix+seg_suffix_use+NII_SUFFIX):
                    suffix_to_nifti[suffix] = nib.load(os.path.join(path_sub_dir, file_type.replace('_seg', '')))

        if '' in suffix_to_nifti.keys():
            mhd_case, fhd_case, rhd_case = calc_statistics(suffix_to_nifti)
            for suffix in suffix_to_nifti.keys():
                MHD[suffix].append(mhd_case[suffix])
                FHD[suffix].append(fhd_case[suffix])
                RHD[suffix].append(rhd_case[suffix])

    # Print results
    print('Results\n')
    for suffix in MHD.keys():
        mhd_array = np.array(MHD[suffix])
        fhd_array = np.array(FHD[suffix])
        rhd_array = np.array(RHD[suffix])

        mhd_mean = np.mean(mhd_array, axis=0)
        fhd_mean = np.mean(fhd_array, axis=0)
        rhd_mean = np.mean(rhd_array, axis=0)

        mhd_std = np.std(mhd_array, axis=0)
        fhd_std = np.std(fhd_array, axis=0)
        rhd_std = np.std(rhd_array, axis=0)

        print("Version - %s" % suffix)
        # Print total result
        print("MHD  = %f, %f ||| %s , %s" % (mhd_mean.mean(), mhd_std.mean(), mhd_mean, mhd_std))
        print("FHD  = %f, %f ||| %s , %s" % (fhd_mean.mean(), fhd_std.mean(), fhd_mean, fhd_std))
        print("RHD = %f, %f ||| %s , %s\n" % (rhd_mean.mean(), rhd_std.mean(), rhd_mean, rhd_std))


def calc_statistics(suffix_to_nifti):
    """
    Get dictionary - suffix to nifti, return statistics
    :param suffix_to_nifti:
    :return:
    """
    ref = suffix_to_nifti[''].get_data()
    ref_binary = ref > 0
    mhd = {}
    fhd = {}
    rhd = {}
    num_of_cls = len(CLASSES)
    for (ver_name, ver_nifti) in suffix_to_nifti.iteritems():
        if ver_name == '':
            mhd[ver_name] = num_of_cls*[1.0]
            fhd[ver_name] = num_of_cls*[0.0]
            rhd[ver_name] = num_of_cls*[1.0]
            continue
        data = ver_nifti.get_data()
        data_binary = data > 0
        mhd_temp = []
        fhd_temp = []
        rhd_temp = []

        for i in range(0, data.shape[-1]):
            # MHD, FHD, RHD = ModHausdorffDist(bw_ref == 0, bw_dat == 0)
            # MHD, FHD, RHD = ModHausdorffDist(ref_binary[:,:,i], data_binary[:,:,i])
            MHD, FHD, RHD = calc_MHD(ref_binary[:,:,i], data_binary[:,:,i])

            mhd_temp.append(MHD)
            fhd_temp.append(FHD)
            rhd_temp.append(RHD)

        mhd[ver_name] = np.array(mhd_temp).mean()
        fhd[ver_name] = np.array(fhd_temp).mean()
        rhd[ver_name] = np.array(rhd_temp).mean()
    return mhd, fhd, rhd


def calc_dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def calc_MHD(brain1, brain2):
    """
    Given two brain images, calculate the modified hausdorff distance
    :param brain1: brain 1 binary image, 1 foreground, 0 background
    :param brain2: brain 2 binary image, 1 foreground, 0 background
    :return:
    """
    # Calculate distance function
    bw_1 = calc_bwdist(brain1)
    bw_2 = calc_bwdist(brain2)

    contour_1 = bw_1 == 0
    contour_2 = bw_2 == 0

    FHD = (1.0 / np.sum(contour_2)) * np.sum(bw_1 * contour_2)
    RHD = (1.0 / np.sum(contour_1)) * np.sum(bw_2 * contour_1)
    MHD = 0.5 * (FHD + RHD)
    return MHD, FHD, RHD


def calc_bwdist(data):
    """
    Given data (binary), return distance function
    :param data:
    :return:
    """
    return 0.5 * (distance_transform_edt(data == 0) + distance_transform_edt(data == 1)) - 0.5


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Go over segmentation results and calc statistics.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='data directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--suffixes', dest='suffixes', default=None, type=str, help='suffixes')
    parser.add_argument('--brain_only', dest='brain_only', default=True, type=bool, help='brain only')
    args = parser.parse_args()

    calc_brain_statistics(args.data_dir, args.num_of_cases, args.suffixes, args.brain_only)