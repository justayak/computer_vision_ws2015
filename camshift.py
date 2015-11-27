from matplotlib import colors
import numpy as np


def camshift(F, Target):
    """

    :param F:
    :param Target:
    :return:
    """
    x = -1
    y = -1
    w = -1
    h = -1

    return x, y, w, h


def probability_matrix(F, Histo):
    """
    Calculates the probability matrix with the given Histogram
    :param F: Frame as RGB matrix
    :param Histo: probability distribution of target element
    :return: probability matrix
    """
    HSV = colors.rgb_to_hsv(F)
    H = np.zeros((HSV.shape[0], HSV.shape[1]), dtype=HSV.dtype)
    H[:, :] = HSV[:, :, 0]
    H *= (len(Histo) - 1)  # normalize for array access
    H = H.astype(int)
    P = np.zeros((HSV.shape[0], HSV.shape[1]), dtype=np.float16)
    h = H.shape[0]
    w = H.shape[1]
    for x in range(0, w):
        for y in range(0, h):
            P[y][x] = Histo[H[y][x]]
    return P


def target_histogram(I):
    """
    Calculates the probability histogram
    :param I:
    :return: 1d probability with 255 elements where the list
        positions represent the H value
    """
    HSV = colors.rgb_to_hsv(I)
    H = np.zeros(HSV.shape[0], HSV.shape[1], dtype=HSV.dtype)
    H[:, :] = HSV[:, :, 0]
    hist, bins = np.histogram(
        H.flatten("C"),
        bins=255,
        range=(0.0, 1.0))
    total = H.shape[0] * H.shape[1]
    return hist/total
