# some computer vision functions
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def paint_mats(mats):
    """
    @mats {Matrices}: images to draw
    """
    cols = len(mats)

    fig, axs = plt.subplots(
    1,
    cols,
    figsize=(32, 16),
    sharex=True,
    sharey=True)

    i = 0
    try:
        for ax in axs:
            ax.axis('off')
            ax.imshow(mats[i], cmap=plt.cm.gray)
            ax.set_adjustable('box-forced')
            i += 1
    except:
        axs.axis('off')
        axs.imshow(mats[i], cmap=plt.cm.gray)
        axs.set_adjustable('box-forced')
            
    plt.show()

def imresize(im, sz):
    pli_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize(sz))

def histeq(im, nbr_bins=256):
    """ histogram eq for grayscale img """
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # linear interpol. to find new pixel values
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def compute_average(imlist):
    """ compute average of a list of images """
    are_filepaths = isinstance(imlist[0], str)

    averageim = imlist[0].copy()
    if are_filepaths:
        averageim = np.array(Image.open(imlist[0]), 'f')
    else:
        averageim = np.uint8(averageim)

    for imname in imlist[1:]:
        try:
            if are_filepaths:
                averageim += np.array(Image.open(imname))
            else:
                averageim += np.uint8(imname)
        except:
            print("skip " + imname)
    averageim = np.float64(averageim)
    averageim /= len(imlist)  # potentially buggy
    return np.array(averageim, 'uint8')

def pca(X):
    """ 
    principle component analysis 
    input: X, matrix with training data stored as flattend arrays in rows
    return: projection matrix (with important dim first)
    """
    num_data, dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        M = np.dot(X, X.T)  # covariance matrix
        e, EV = np.linalg.eigh(M)  # eigenval and eigenvec
        tmp = np.dot(X.T, EV)  # compact trick
        V = tmp[::-1]
        S = np.sqrt(e)[::-1]  # reverse, eigenvec are in incr. order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]  # return first num_data

    # return projection mat & var & mean
    return V, S, mean_X
