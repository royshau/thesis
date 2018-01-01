import numpy as np
from numpy.fft import *
from scipy.interpolate import griddata
from scipy.misc import imresize
import math
EPS = 0.000001

def get_dummy_k_space_and_image(img):
    """
    Get an MRI image and return k_space and dummy image
    :param img: 2d numpy.array or 3d if dim=3
    :return: k_space, img
    """

    k_space = (np.zeros_like(img) + 0j).astype('complex64')
    dummy_img = np.zeros_like(img)

    if len(img.shape) == 2:
        # Create k-space from image, assuming this is the fully-sampled k-space.
        k_space = fft2c(img).astype('complex64')

        # Clipping
        k_space.real[np.where(np.abs(k_space.real) < EPS)] = 0
        k_space.imag[np.where(np.abs(k_space.imag) < EPS)] = 0

        # Reconstruct image
        dummy_img = np.abs(ifft2c(k_space)).astype(np.float32)
    else:
        for i in range(0, img.shape[2]):
            # Create k-space from image, assuming this is the fully-sampled k-space.
            k_space[:,:,i] = (fft2c(img[:,:,i])).astype('complex64')

            # Clipping
            # print np.sum(np.abs(k_space[:, :, i].real) < EPS) / float(len(k_space[:, :, i].real.ravel()))
            k_space[:,:,i].real[np.where(np.abs(k_space[:,:,i].real) < EPS)] = 0
            k_space[:,:,i].imag[np.where(np.abs(k_space[:,:,i].imag) < EPS)] = 0

            # Reconstruct image
            dummy_img[:, :, i] = np.abs(ifft2c(k_space[:, :, i])).astype(np.float32)
            # print np.sum(np.abs(dummy_img[:, :, i]) < EPS) / float(len(dummy_img[:, :, i].ravel()))

            dummy_img[:,:,i][np.where(np.abs(dummy_img[:,:,i]) < EPS)] = 0
            # dummy_img[:,:,i][np.where(dummy_img[:,:,i] > 1)] = 1

    return k_space, dummy_img


def get_image_from_kspace(k_real, k_imag):
    """
    Return image from real and imaginary k-space values
    :param k_real:
    :param k_imag:
    :return:
    """
    k_space = (np.zeros_like(k_real) + 0j).astype('complex64')
    ret_img = np.zeros_like(k_real)

    if len(k_real.shape) == 2:
        # Create k-space from real and imaginary part
        k_space.real = k_real
        k_space.imag = k_imag
        ret_img = np.abs(ifft2c(k_space)).astype(np.float32)

    else:
        for i in range(0, k_real.shape[0]):
            # Create k-space from image, assuming this is the fully-sampled k-space.
            k_space[i,:,:] = (np.zeros_like(k_real[i,:,:]) + 0j).astype('complex64')
            k_space[i,:,:].real = k_real[i,:,:]
            k_space[i,:,:].imag = k_imag[i,:,:]

            # Reconstruct image
            ret_img[i, :, :] = np.abs(ifft2c(k_space[i, :, :])).astype(np.float32)

    return ret_img


def fft2c(x):
    """
    res = fft2c(x)
    orthonormal forward 2D FFT
    (c) Michael Lustig 2005
    :param x: numpy array (image)
    :return: orthonormal 2D fft
    """
    norm_factor = 1.0 / np.sqrt(len(x.ravel()))
    ret = norm_factor * fftshift(fft2(ifftshift(x)))
    return ret


def ifft2c(x):
    """
    res = ifft2c(x)
    orthonormal forward 2D IFFT
    (c) Michael Lustig 2005
    :param x: numpy array (image)
    :return: orthonormal 2D ifft
    """
    norm_factor = np.sqrt(len(x.ravel()))
    ret = norm_factor * ifftshift(ifft2(fftshift(x)))
    return ret


def zero_padding(k_space_real, k_space_imag, mask):
    """
    return the image with the same dims as img_input.
    put zero in mask
    :param k_space_real
    :param k_space_imag
    :param mask: binary mask
    """
    out_real = np.zeros_like(mask)
    out_imag = np.zeros_like(mask)

    filled = 0
    for line in range(mask.shape[1]):
        if all(mask[line, : ]>0):
            out_real[line, :] = k_space_real[filled, :]
            out_imag[line, :] = k_space_imag[filled, :]
            filled += 1

    assert filled == k_space_real.shape[0]
    return out_real, out_imag


def interpolated_missing_samples(img_input, dims_out, method):
    """
    return the image with the same dims as img_input.
    Interpolated missing samples
    :param img_input: input image
    :param dims_out: output dimension tuple
    :param method: method of interpolation
    """
    if len(img_input.shape) > 2:
        # For 3D volume
        ret = np.zeros_like(img_input)
        for i in range(0, img_input.shape[0]):
            ret[i,:,:] = imresize(arr=img_input[i,:,:], size=dims_out, interp=method, mode=None)
    else:
        ret = imresize(arr=img_input, size=dims_out, interp=method, mode=None)
    return ret


def pad_image_with_zeros_square(dat):
        """
        Padding image with zeros
        :return:
        """

        shap = np.array(dat.shape)
        max_shape = shap.max()
        if shap[0] == shap[1] == max_shape:
            ret = dat
        else:
            ret = np.zeros((max_shape, max_shape, shap[2]))
            if shap[0] < max_shape:
                min_shape = shap[0]
                min_idx = 0
            else:
                min_shape = shap[1]
                min_idx = 1
            missing_lines = max_shape - min_shape

            lines_from_top = np.floor(0.5*missing_lines).astype(np.int)
            lines_from_bottom = (missing_lines - lines_from_top).astype(np.int)
            if min_idx == 0:
                ret[lines_from_top:-lines_from_bottom, :, :] = dat
            else:
                ret[:, lines_from_top:-lines_from_bottom, :] = dat
        return ret


def pad_image_with_zeros_fixed(dat, to_size=[256, 180]):
    """
    Given a data, this function pad with zeros samller dimensions
    :param dat:
    :param fixed_size:
    :return:
    """
    shap = np.array(dat.shape)
    to_size = np.array(to_size)
    if shap[0] == to_size[0] and shap[1] == to_size[1]:
        ret = dat
    else:
        ret = np.zeros((to_size[0], to_size[1], shap[2]))
        missing_lines = [0,0]
        lines_from_top = [0,0]
        lines_from_bottom = [0,0]
        for s in [0,1]:
            if shap[s] < to_size[s]:
                missing_lines[s] = to_size[s] - shap[s]
                lines_from_top[s] = np.floor(0.5 * missing_lines[s]).astype(np.int)
                lines_from_bottom[s] = (missing_lines[s] - lines_from_top[s]).astype(np.int)
            # if s == 0:
        if lines_from_bottom[0] == 0:
            ret[lines_from_top[0]:, lines_from_top[1]:-lines_from_bottom[1], :] = dat
        elif lines_from_bottom[1] == 0:
            ret[lines_from_top[0]:-lines_from_bottom[0], lines_from_top[1]:, :] = dat
        else:
            ret[lines_from_top[0]:-lines_from_bottom[0], lines_from_top[1]:-lines_from_bottom[1], :] = dat
            # else:
            #     ret[:, lines_from_top[0]:-lines_from_bottom[0], :] = dat
    return ret
