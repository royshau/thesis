import numpy as np
import numpy.fft as fft
from scipy import interpolate as inter
import pywt

def SoftThresh(x, lamb):
    xhat = np.zeros_like(x)
    eps = 2e-16
    xhat = ((np.absolute(x) - lamb) / (np.absolute(x) + eps)) * x * (np.absolute(x) > lamb)
    # print (xhat)
    return xhat


def fft2c(x):
    N = x.shape[0] * x.shape[1]
    return 1 / np.sqrt(N) * fft.fftshift(fft.fft2(fft.ifftshift(x)))


def ifft2c(x):
    N = x.shape[0] * x.shape[1]
    return np.sqrt(N) * fft.fftshift(fft.ifft2(fft.ifftshift(x)))


def wavelet_thresh(img, threshold, level, wavtype='db2'):
    coeffs2 = pywt.wavedec2(img, wavtype, level=level)
    NewWaveletCoeffs = map(lambda x: SoftThresh(x, threshold), coeffs2)
    return pywt.waverec2(NewWaveletCoeffs, wavtype)


def mask_and_fill(data, mask, method=None):
    # This Function masks data and interpolates missing samples.
    # valid method = 'zeros' 'cubic' 'linear' 'None'
    # None returns masked data. 'zeros' returns normalized masked data.
    if method is None:
        m_data = data * mask
        return m_data
    elif method == 'zeros':
        m_data = data * mask
        return m_data
    elif method == 'cubic' or method == 'linear' or method == 'nearest':
        # Create values and mesh to fill:
        x = np.arange(0, data.shape[1])
        y = np.arange(0, data.shape[0])
        data_nan = data.copy()
        data_nan[mask == 0] = np.nan;
        xx, yy = np.meshgrid(x, y)
        masked_data = np.ma.masked_invalid(data_nan)
        data_points_x = xx[~masked_data.mask]
        data_points_y = yy[~masked_data.mask]
        data_vals = masked_data[~masked_data.mask]
        real_vals = np.real(data_vals)
        imag_vals = np.imag(data_vals)
        # Interpolate:
        real_inter = inter.griddata((data_points_x, data_points_y), real_vals.ravel(),
                                    (xx, yy),
                                    method=method, fill_value=0)
        imag_inter = inter.griddata((data_points_x, data_points_y), imag_vals.ravel(),
                                    (xx, yy),
                                    method=method, fill_value=0)
        return real_inter + 1j * imag_inter
    else:
        print('Invalid interpolation method. returning NaN')
        return np.nan

def psnr(img,ref):
  max_pix = np.max(ref)
  mse = np.mean((img - ref) ** 2)
  return 10*np.log10((max_pix**2)/mse)