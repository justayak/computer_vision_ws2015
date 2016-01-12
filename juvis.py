# some computer vision functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
import math

def vectorize_images(images):
    """ flattens all images, each image is a row """
    return np.array([im.copy().flatten() for im in images])

def unify_images(folder_str, size):
    files = [f for f in listdir(folder_str) if isfile(join(folder_str, f))]
    size = size, size
    for f in files:
        if f.endswith('png'):
            full_name = join(folder_str, f)
            thmb_name = full_name + ".thb"
            im = Image.open(full_name)
            w, h = im.size
            if w != h:
                # not squared: square it!
                oldsize = str(im.size)
                if w > h:
                    cropLeft = int(math.floor((w - h) / 2.0))
                    cropRight = w - int(math.ceil((w - h)/2.0))
                    im = im.crop((cropLeft, 0, cropRight, h))
                else:
                    cropTop = int(math.floor((h - w)/2.0))
                    cropBottom = h - int(math.ceil((h - w)/2.0))
                    im = im.crop((0, cropTop, w, cropBottom))
                print("resize to: " + str(im.size) + " from " + oldsize)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(thmb_name, "PNG")

def load_images_as_mat(folder_str, extension="png", gray=False):
    """
    tbd
    """
    files = [join(folder_str, f) for f in listdir(folder_str) 
            if isfile(join(folder_str, f)) and f.endswith(extension)]
    mats = []
    for f in files:
        if gray:
            mats.append(np.array(Image.open(f).convert('L')))
        else:
            mats.append(np.array(Image.open(f)))
    return mats

def plot2d(M):
    """
    @M {np.array}
    Plots the 2d matrix with no interpolation.
    The higher the value, the brighter the pixel
    """
    plt.imshow(M, cm.gray, interpolation="nearest")
    plt.axis("off")
    plt.show()

def normalize(X, Y):
    N = np.sqrt(X**2 + Y**2)
    Z = (N == 0) * 1
    N += Z
    Xn = X/N
    Yn = Y/N
    return Xn, Yn

def rad2pos(R):
    """
    @R {np.array} Matrix in radians
    """
    X = np.zeros(R.shape)
    Y = np.zeros(R.shape)
    

    
    return normalize(X, Y)

def plot_vec(u, v):
    """
    """
    # flip top-bottom as we want the origin to be top/left
    u = np.flipud(u)
    v = np.flipud(v)
    u, v = normalize(u, v)
    height = u.shape[0]
    width = v.shape[1]
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x,y)
    plt.quiver(X,Y,u,v, pivot='middle', angles='uv', headlength=6)
    plt.axis('off')
    plt.xlim(-1, width)
    plt.ylim(-1, height)
    plt.show()

def paint_mats(mats, interpolation='bilinear', vmin=None, vmax=None, max_row=10):
    """
    @mats {Matrices}: images to draw
    """
    cols = len(mats)
    rows = 1

    MAX_ROW = max_row
    
    if cols > MAX_ROW:
        rows = int(math.ceil(cols/MAX_ROW))
        cols = MAX_ROW

    size = (32, 32)

    fig, axs = plt.subplots(
    rows,
    cols,
    figsize=size,
    sharex=True,
    sharey=True)

    i = 0
    try:
        for ax in axs:
            if hasattr(ax, '__len__'):
                for ax_sub in ax:
                    ax_sub.axis('off')
                    if len(mats) > i:
                        ax_sub.imshow(
                            mats[i],
                            cmap=plt.cm.gray,
                            vmin=vmin,
                            vmax=vmax,
                            interpolation=interpolation)
                        ax_sub.set_adjustable('datalim')
                        i += 1
            else:
                ax.axis('off')
                ax.imshow(
                    mats[i],
                    cmap=plt.cm.gray,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation=interpolation)
                ax.set_adjustable('box-forced')
                i += 1
    except:
        axs.axis('off')
        axs.imshow(
            mats[i],
            cmap=plt.cm.gray,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation)
        axs.set_adjustable('box-forced')
            
    plt.show()

def imresize(im, sz):
    pil_im = Image.fromarray(np.uint8(im))
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

    if dim > num_data and False:
        M = np.dot(X, X.T)  # covariance matrix
        e, EV = np.linalg.eigh(M)  # eigenval and eigenvec
        tmp = np.dot(X.T, EV).T  # compact trick
        V = tmp[::-1]
        print(M.shape)
        print(M)
        S = np.sqrt(e)[::-1]  # reverse, eigenvec are in incr. order
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        # PCA - SVD used
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]  # return first num_data

    # return projection mat & var & mean
    return V, S, mean_X
