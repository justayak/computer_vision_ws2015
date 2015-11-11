import numpy as np
from scipy import ndimage, stats
from matplotlib import pyplot as plt, colors

I = ndimage.imread("racecar.png")

I = I[260:360, 460:660]

HSV = colors.rgb_to_hsv(I)

H = np.zeros((HSV.shape[0], HSV.shape[1]), dtype=HSV.dtype)
H[:, :] = HSV[:, :, 0]
plt.hist(H.flatten('C'), bins=255, range=(0.0, 1.0))
plt.show()