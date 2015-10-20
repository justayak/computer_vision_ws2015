from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np

def show_mat(mat, gray=False):
    if (gray):
        plt.imshow(mat, cmap=plt.get_cmap('gray'))
    else:
        plt.imshow(mat)
    plt.show()


# Erste Aufgabe
I = ndimage.imread("image.jpg")
#show_mat(I)

# Zweite Aufgabe
S = I[50:110, 110:170]
#show_mat(S)

# Dritte Aufgabe
R = I.copy()
R[:,:,1] *= 0
R[:,:,2] *= 0
#show_mat(R)

# Vierte Aufgabe
#F = np.fliplr(np.flipud(I))
F = np.rot90(I, 2)
#show_mat(F)

# Fuenfte Aufgabe
G = ndimage.imread("image.jpg", flatten=True)
G = np.absolute(G - 255)
#G = np.invert(G)
show_mat(G, gray=True)

