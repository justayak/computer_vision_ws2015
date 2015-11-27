import numpy as np
from scipy import ndimage
import cv2
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
    H *= (len(Histo) - 1)  # normalize for array access
    H = H.astype(int)
    P = np.zeros((HSV.shape[0], HSV.shape[1]), dtype=np.float16)
    h = H.shape[0]
    w = H.shape[1]
    for x in range(0, w):
        for y in range(0, h):
            P[y][x] = Histo[H[y][x]]
    return P

def histo(I):
    """
    Calculates the probability histogram
    :param I:
    :return: 1d probability histogram with 255 elements
    """
    HSV = colors.rgb_to_hsv(I)
    H = np.zeros((HSV.shape[0], HSV.shape[1]), dtype=HSV.dtype)
    H[:, :] = HSV[:, :, 0]
    hist, bins = np.histogram(H.flatten("C"), bins=255, range=(0.0, 1.0))
    total = H.shape[0] * H.shape[1]
    return hist/total

def get_nth_frame(video_file, n):
    cam = cv2.VideoCapture(video_file)
    if cam.isOpened():
        for x in range(0, n+1):
            _, img = cam.read()
            if x == n:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img

def main():
    # Calculate CAR histogram
    I = ndimage.imread("racecar.png")
    Car = I[270:350, 480:640]
    #Car = I[260:360, 460:660]
    F = get_nth_frame("racecar.avi", 300)
    BLUE_CAR_HISTO = histo(Car)
    P = camshift(F, BLUE_CAR_HISTO)

    plt.imshow(P, cmap='gray', vmin=0, vmax=0.2)
    #plt.imshow(Car)
    plt.show()
    # apply histogram

main()
