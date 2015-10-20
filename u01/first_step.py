from scipy import (ndimage, misc)
from matplotlib import pyplot as plt

I = ndimage.imread("image.jpg")

plt.imshow(I, interpolation='nearest')
plt.show()
