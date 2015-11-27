from matplotlib import pyplot as plt
import numpy as np
import cv2


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
        #stream.append(img)
        while ret:
            ret, img = cam.read()
            if ret:
                stream.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                #stream.append(img)
    return stream


def play_video(video):
    frame = video[0]
    frames = len(video)
    plt.axis("off")
    im = plt.imshow(frame)
    for i in range(1, frames):
        im.set_data(video[i])
        plt.pause(0.02)
    plt.show()