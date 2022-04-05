# Histogram Equalization
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Comparing the results of Historgam Equalization and Adaptive
#              Histogram Equalization for a stram of images


import cv2
import numpy as np
import os


def histEqual(frame):
    """
    Definition
    ---
    Method to perform histogram equalisation on a 2D intensity array

    Parameters
    ---
    frame : 2D array

    Returns
    ---
    frame : 2D array after equalisation
    """
    hist = np.zeros(256)
    hist, bins = np.histogram(frame.flatten(), 256, [0, 255])
    hist = hist.cumsum()
    hist = hist / hist[-1]
    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            frame[x, y] = 255 * hist[frame[x, y]]
    return frame


def adaptiveHistEqual(frame, t_no=7):
    """
    Definition
    ---
    Method to perform adaptive histogram equalisation on a 2D intensity array

    Parameters
    ---
    frame : 2D array
    t : number of tiles per axis (default = 7)

    Returns
    ---
    frame : 2D array after equalisation
    """
    img_mod = np.zeros_like(frame)
    k = (frame.shape[0]//t_no, frame.shape[1]//t_no)
    for i in range(0, frame.shape[0] - k[0], k[0]):
        for j in range(0, frame.shape[1] - k[1], k[1]):
            img_mod[i:i+k[0], j:j+k[1]] = histEqual(frame[i:i+k[0], j:j+k[1]])
    return img_mod


if __name__ == '__main__':
    path = 'res/adaptive_hist_data'
    files = os.listdir(path)
    files = sorted(files)
    for f in files:
        img = cv2.imread("res/adaptive_hist_data/" + f)
        cv2.imshow('input', img)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = np.array(cv2.split(ycrcb))
        channels[0] = histEqual(channels[0])
        ycrcb = cv2.merge(channels)
        img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        cv2.imshow('histogram equalisation', img)
        channels[0] = adaptiveHistEqual(channels[0])
        ycrcb = cv2.merge(channels)
        img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        cv2.imshow('adaptive histogram equalisation', img)
        cv2.waitKey(10)
