import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

I = np.loadtxt("pointdata_3d.csv", delimiter=",")
u = I[0]
v = I[1]
w = I[2]

f = 10

def exec(fun):
    return np.array([fun(i) for i in range(2000)])

X = exec(lambda i: f*u[i]/w[i])
Y = exec(lambda i: f*v[i]/w[i])
#plt.plot(X, Y)

f /= 2
X = exec(lambda i: f*u[i]/w[i])
Y = exec(lambda i: f*v[i]/w[i])
#plt.plot(X, Y)

X = exec(lambda i: f*u[i])
Y = exec(lambda i: f*v[i])
plt.plot(X, Y)
plt.show()
