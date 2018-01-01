import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# plt.style.use('ggplot')

psnr_no_masking = {
    "2.5": {
        "Zero-filled": [32.044, 2.616],
        "CS-MRI": [39.053, 2.479],
        "IM-CNN-L2": [35.281, 2.210],
        "CNN-L2": [38.978, 2.454],
        "Proposed": [39.802, 2.489],
    },
    "4": {
        "Zero-filled": [26.447, 1.997],
        "CS-MRI": [33.273, 2.452],
        "IM-CNN-L2": [31.577, 2.109],
        "CNN-L2": [33.405, 2.232],
        "Proposed": [34.595, 2.519],
    },
    "6": {
        "Zero-filled": [16.470, 2.181],
        "CS-MRI": [26.951, 3.380],
        "IM-CNN-L2": [23.417, 2.676],
        "CNN-L2": [31.010, 2.299],
        "Proposed": [31.555, 2.487],
    }
}

# example data
methods = psnr_no_masking['2.5'].keys()
x = np.array([2.5, 4, 6])
y_mean = {}
y_std = {}
for method in methods:
    y_mean[method] = [psnr_no_masking["2.5"][method][0], psnr_no_masking["4"][method][0], psnr_no_masking["6"][method][0]]
    y_std[method]  = [0.5*psnr_no_masking["2.5"][method][1], 0.5*psnr_no_masking["4"][method][1], 0.5*psnr_no_masking["6"][method][1]]

# First illustrate basic pyplot interface, using defaults where possible.
plt.figure()
fmts = ['D', '+', '|', 'x', 'o']
ind = 0
ad = 0
for method in methods:
    plt.errorbar(x+ad, y_mean[method], yerr=y_std[method], fmt=fmts[methods.index(method)], label=method, capsize=5, capthick=1.5)
    ad += 0.05

plt.title("Error in PSNR, without masking")
plt.grid('on', linestyle='-', linewidth=0.2)
plt.ylabel('PSNR[dB]')
plt.xlabel('Sampling factor')
plt.legend()
plt.xticks(x, x)
plt.show()