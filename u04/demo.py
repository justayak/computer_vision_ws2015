import numpy as np
import cv2
from matplotlib import pyplot as plt


def imshow(frame):
    im = cv2.cvtColor(frame[r:r+h, c:c+w], cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.show()


cap = cv2.VideoCapture("../u03/racecar.avi")

ret, frame = cap.read()

r, h, c, w = 260, 90, 470, 165
track_window = (c, r, w, h)

roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(1):
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        #else:
            #cv2.imwrite(chr)
    else:
        break

cv2.destroyAllWindows()
cap.release()

