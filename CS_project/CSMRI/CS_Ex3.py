import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.io as imp

def fft2c(x):
    N = x.shape[0]*x.shape[1]
    print N
    return 1/np.sqrt(N)*fft.fftshift(fft.fft2(fft.ifftshift(x)))

def ifft2c(x):
    N = x.shape[0]*x.shape[1]
    return np.sqrt(N)*fft.fftshift(fft.ifft2(fft.ifftshift(x)))

mat = imp.loadmat('brain.mat')
brain = np.array(mat['im'])
print(fft2c(brain).shape)
plt.imshow(fft2c(brain),cmap='gray')
plt.show()