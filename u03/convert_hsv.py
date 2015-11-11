import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

I = ndimage.imread("racecar.png")

Idash = I[500:505, 50:55]

Idash[0][0][0] = 0
Idash[0][0][1] = 255
Idash[0][0][2] = 0

Idash[4][0][0] = 0
Idash[4][0][1] = 0
Idash[4][0][2] = 255

# convert to HSV
Idash = Idash.copy()/255

R = np.zeros((Idash.shape[0], Idash.shape[1]), dtype=Idash.dtype)
G = np.zeros((Idash.shape[0], Idash.shape[1]), dtype=Idash.dtype)
B = np.zeros((Idash.shape[0], Idash.shape[1]), dtype=Idash.dtype)
R[:, :] = Idash[:, :, 0]
G[:, :] = Idash[:, :, 1]
B[:, :] = Idash[:, :, 2]
stacked = np.dstack((R, G, B))
Cmax = stacked.max(2)
Cmin = stacked.min(2)
Delta = Cmax - Cmin

print(Delta)



#plt.imshow(Idash)
#plt.show()
