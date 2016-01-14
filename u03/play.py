from scipy import signal, ndimage
import numpy as np
from matplotlib import pyplot as plt

I = ndimage.imread("racecar.png")


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

I = rgb2gray(I)
Sx = ndimage.sobel(I, 0)

print(Sx.shape)

F = np.zeros((Sx.shape[0], Sx.shape[1], 3))
F[...,:3] = Sx.reshape((Sx.shape[0], Sx.shape[1], 1))

plt.imshow(F)
plt.show()