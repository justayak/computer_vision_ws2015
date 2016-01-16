from scipy import signal, ndimage
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

I = ndimage.imread("racecar.png")

Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

I = rgb2gray(I)
#Sx = ndimage.sobel(I, 0)
Sx = signal.convolve2d(I, Kx, boundary='symm', mode='same')

#F = np.zeros((Sx.shape[0], Sx.shape[1], 3))
#F[...,:3] = Sx.reshape((Sx.shape[0], Sx.shape[1], 1))

plt.imshow(Sx, cmap=cm.Greys_r)
plt.show()
