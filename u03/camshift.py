import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt, colors

def camshift(F, Histo):
    """
    Calculates the probability matrix with the given Histogram
    :param F: Frame as RGB matrix
    :param Histo: probability distribution of target element
    :return: probability matrix
    """
    HSV = colors.rgb_to_hsv(F)
    H = np.zeros((HSV.shape[0], HSV.shape[1]), dtype=HSV.dtype)
    H[:, :] = HSV[:, :, 0]
    P = np.zeros((HSV.shape[0], HSV.shape[1]), dtype=HSV.dtype)
    return P

# Calculate CAR histogram
I = ndimage.imread("racecar.png")
I = I[260:360, 460:660]

HSV = colors.rgb_to_hsv(I)
H = np.zeros((HSV.shape[0], HSV.shape[1]), dtype=HSV.dtype)
H[:, :] = HSV[:, :, 0]

BLUE_CAR_HISTO, bins = np.histogram(H.flatten("C"), bins=255, range=(0.0, 1.0))

# apply histogram