import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2

S = ndimage.imread("needle.png")  # Symbol
H = ndimage.imread("heystack.png")  # Heystack

def hu_moments(I):
    return cv2.HuMoments(cv2.moments(I)).flatten()

moments = hu_moments(S)


ret, thresh = cv2.threshold(H, 1, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

bboxes = []


def insert_bbx(bboxes, bb):
    """
    Make sure that no bounding box is added that is part of another one
    :param bboxes:
    :param bb:
    :return: True when bb is ok, False otherwise
    """
    x = bb[0]
    y = bb[1]
    for c_bb in bboxes:
        c_x = c_bb[0]
        c_y = c_bb[1]
        c_X = c_x + c_bb[2]
        c_Y = c_y + c_bb[3]
        if c_x < x < c_X:
            if c_y < y < c_Y:
                return False
    bboxes.append(bb)
    return True

idx = 0
for cnt in contours:
    idx += 1
    x, y, w, h = cv2.boundingRect(cnt)
    x -= 5
    y -= 5
    w += 10
    h += 10
    if insert_bbx(bboxes, (x, y, w, h)):
        cv2.rectangle(im2, (x, y), (x+w, y+h), (200, 0, 0), 2)

assert len(bboxes), 4


def extract(I, bb):
    """
    :param I: image
    :param bb: bounding box
    :return: extract image with bounding box
    """
    x, y, w, h = bb[0], bb[1], bb[2], bb[3]
    print(bb)
    return I[y:y+h, x:x+w]


A = extract(H, bboxes[0])
B = extract(H, bboxes[1])
C = extract(H, bboxes[2])
D = extract(H, bboxes[3])

m_A = hu_moments(A)
m_B = hu_moments(B)
m_C = hu_moments(C)
m_D = hu_moments(D)


def difference(L, R):
    return np.sum(np.subtract(L, R))


a = difference(moments, m_A)
b = difference(moments, m_B)
c = difference(moments, m_C)
d = difference(moments, m_D)

min_pos = np.argmin([a, b, c, d])
print(min_pos)

plt.imshow(C)
plt.show()



