import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

I = ndimage.imread("racecar.png");

# convert to HSV
R = I.copy()
R = I/255
print(str(R))


plt.imshow(I)
plt.show()
