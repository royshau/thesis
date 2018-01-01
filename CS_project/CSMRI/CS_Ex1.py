import numpy as np
import matplotlib.pyplot as plt
x = np.append([np.zeros([128-5,1])],[np.linspace(1,5,5) /5])
x=np.random.permutation(x)
y = x + 0.05*np.random.randn(128)
print(y)
plt.figure(1)
plt.stem(y)
dic=(0.01,0.05,0.1,0.2)
lamb = 0.1

x1 = (y-lamb)*(y>lamb) + (y+lamb)*(y<-lamb)
print (x1)