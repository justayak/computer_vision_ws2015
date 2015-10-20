from scipy import ndimage
from matplotlib import pyplot as plt

I = ndimage.imread("image.jpg")

S = I[50:110, 110:170]

plt.imshow(S, interpolation='nearest')
plt.show()
