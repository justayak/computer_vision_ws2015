from matplotlib import pyplot as plt
import cv2
import numpy as np


def load_video_as_rgb(video_file):
    """
    Load a video into a python array as RGB images
    :param video_file:
    :return: List of frames in RGB
    """
    cam = cv2.VideoCapture(video_file)
    stream = []
    if cam.isOpened():
        ret, img = cam.read()
        stream.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        while ret:
            ret, img = cam.read()
            if ret:
                stream.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return stream


def play_video(video, overlay=None):
    """
    Plays a list of rgb images
    careful, performance is not key here!
    :param video: List of rgb images
    :param overlay: List of tuple for drawing rect per frame
            (x, y, w, h)
    :return:
    """
    if overlay is not None:
        assert(len(video) == len(overlay))

    frame = video[0]  # i dont care for the first overlay frame..
    frames = len(video)
    plt.axis("off")
    RED = np.array([255, 0, 0])
    LW = 6

    ax = plt.gca()
    im = ax.imshow(frame)
    for i in range(1, frames):
        frame = video[i]
        if overlay is not None:
            (x, y, w, h) = overlay[i]
            frame[y:y+h, x-LW:x] = RED
            frame[y:y+h, x+w:x+w+LW] = RED
            frame[y+h:y+h+LW, x:x+w] = RED
            frame[y-LW:y, x:x+w] = RED
        im.set_data(frame)
        plt.pause(0.02)
    plt.show()