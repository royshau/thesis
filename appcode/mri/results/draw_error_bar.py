import matplotlib.pyplot as plt
import numpy as np

plt.figure(1)
y_50 = np.array([32.57, 39.0, 39.0])
v_50 = np.array([1.85, 2.48, 2])
y_50r = np.array([30.80, 35.0, 36.0])
v_50r = np.array([1.83, 2.63, 2.49])
y_25 = np.array([27.44, 32.59, 33.6])
v_25 = np.array([1.60, 2.52, 2.2])
x_val = [0, 1, 2]
x_val2 = [0.2, 1.2, 2.2]
x_val3 = [0.4, 1.4, 2.4]
labels = ['Zero-filled ', 'CS-MRI', 'Proposed']
plt.style.use('ggplot')
a =plt.errorbar(x_val3, y_50,fmt='s',color='b' , ecolor='b',capsize=5,capthick=1.5, yerr=v_50.T)[-1][0].set_linestyle('--')
b =plt.errorbar(x_val2, y_50r,fmt='s',color='g' , ecolor='g',capsize=5,capthick=1.5, yerr=v_50r.T)[-1][0].set_linestyle('--')
c =plt.errorbar(x_val, y_25,fmt='s',color='k' , ecolor='k',capsize=5,capthick=1.5, yerr=v_25.T)[-1][0].set_linestyle('--')
plt.xticks(x_val2, labels)
plt.xlim(-0.2,3.2)
plt.ylim(25.0, 42.0)
plt.margins(1)
plt.ylabel("PSNR[dB]")
plt.show()

plt.figure(2)

y_50 = np.array([83.7, 91, 91.5])
v_50 = np.array([10.6, 5.44, 5.5])
y_50r = np.array([78.3, 85, 87])
v_50r = np.array([10, 10, 8])
y_25 = np.array([70.86, 81.7, 84])
v_25 = np.array([10, 10, 9])
x_val = [0, 1, 2]
x_val2 = [0.2, 1.2, 2.2]
x_val3 = [0.4, 1.4, 2.4]
labels = ['Zero-filled ', 'CS-MRI', 'Proposed']
plt.style.use('ggplot')
a = plt.errorbar(x_val3, y_50,fmt='s',color='b' , ecolor='b',capsize=5,capthick=1.5, yerr=v_50.T)[-1][0].set_linestyle('--')
plt.errorbar(x_val2, y_50r,fmt='s',color='g' , ecolor='g',capsize=5,capthick=1.5, yerr=v_50r.T)[-1][0].set_linestyle('--')
plt.errorbar(x_val, y_25,fmt='s',color='k' , ecolor='k',capsize=5,capthick=1.5, yerr=v_25.T)[-1][0].set_linestyle('--')

plt.xticks(x_val2, labels)
plt.xlim(-0.2,3.2)
plt.ylim(55.0, 100.0)
plt.margins(1)
plt.ylabel("DICE[%]")
plt.show()