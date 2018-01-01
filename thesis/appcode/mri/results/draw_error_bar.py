import matplotlib.pyplot as plt
import numpy as np
y = np.array([30.48, 36.78, 37.12, 37.95]) 
v = np.array([0.13, 0.4, 0.82, 0.6])
x_val = [0, 1, 2 , 3]
labels = ['Zero-filled ', 'CS-MRI', 'CNN-L2', 'Proposed']
plt.style.use('ggplot')
plt.errorbar(x_val, y,fmt='s',color='b' , ecolor='r',capsize=5,capthick=1.5, yerr=v.T)
plt.xticks(x_val, labels)
plt.xlim(-0.2,3.2)
plt.ylim(30.0, 40.0)
plt.margins(1)
plt.ylabel("PSNR[dB]")
plt.show()
