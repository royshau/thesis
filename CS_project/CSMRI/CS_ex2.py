import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt


def cfft(x,dim):
    return 1/np.sqrt(x.shape[dim-1])*fft.fftshift(fft.fft(fft.ifftshift(x,dim-1),axis=dim-1),dim-1)

def icfft(x,dim):
    dim=dim-1
    return np.sqrt(x.shape[dim])*fft.fftshift(fft.ifft(fft.ifftshift(x,dim),axis=dim),dim)

def SoftThresh(x,lamb):
    xhat =   np.zeros_like(x)
    eps=2e-16
    xhat = ((np.absolute(x)-lamb)/(np.absolute(x)+eps))*x*(np.absolute(x) > lamb)
    #print (xhat)
    return xhat

x = np.append([np.zeros([128-5,1])],[np.linspace(1,5,5) /5])
x=np.random.permutation(x)
X = cfft(x,1)
Xu = np.zeros([128],dtype=complex)
Xbool = np.linspace(0,127,128)
prm = np.random.permutation(Xbool)[0:31]
Xbool = Xbool[Xbool.astype(int)%4<1]
Xu[prm.astype(int)] = X [prm.astype(int)]
xu_org = icfft(Xu,1)

iters=300
err = np.zeros(iters)
xu = icfft(Xu, 1)
plt.figure(5)
plt.ion()
for i in xrange(iters):
    xu=icfft(Xu,1)
    err[i] = np.mean(np.absolute(x-xu))
    xu=SoftThresh(xu,0.01)
    Xu=cfft(xu,1)
    Xu[prm.astype(int)] = X[prm.astype(int)]
    if i % 10 ==0:
        plt.figure(5)
        plt.clf()
        plt.stem(xu.real,'b')
        plt.stem(xu.imag,'g')
        plt.pause(0.01)
plt.ioff()
plt.figure(1)
plt.title('original x')
plt.stem(x)
plt.figure(2)
plt.title('random sampled x')
plt.stem(np.absolute(xu_org))
#plt.figure(3)
#plt.title('error')
plt.figure(4)
plt.title('recovered x')
plt.stem(np.absolute(xu))
plt.show()
