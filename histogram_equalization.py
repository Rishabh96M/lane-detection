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
    hist = np.zeros(256)
    for row in frame:
        for pix in row:
            hist[pix] = hist[pix] + 1
    # for i in range(1, 256):
    #     hist[i] = hist[i] + hist[i - 1]
    hist = hist.cumsum()
    hist = hist / hist[-1]
    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            frame[x, y] = 255 * hist[frame[x, y]]
    return frame


def adaptiveHistEqual(frame):
    img_mod = np.zeros_like(frame)
    for i in range(frame.shape[0] - 30):
        for j in range(frame.shape[1] - 30):
            kernel = frame[i:i+30, j:j+30]
            for x in range(0, 30):
                for y in range(0, 30):
                    rank = 0
                    for m in range(0, 30):
                        for n in range(0, 30):
                            if(kernel[x, y] > kernel[m, n]):
                                rank = rank + 1
                    img_mod[i, j] = ((rank * 255)/900)
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
        cv2.imshow('output', img)
        cv2.waitKey(0)
