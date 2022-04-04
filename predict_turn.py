# Predict Turn
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Predict the radius of curvature of the lane

import cv2
import numpy as np


def removeBackNoise(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_range = np.array(([[5, 80, 100], [30, 255, 255]]), np.uint8)
    mask_yellow = cv2.inRange(hsv, yellow_range[0], yellow_range[1])

    white_range = np.array(([[0, 0, 210], [255, 45, 255]]), np.uint8)
    mask_white = cv2.inRange(hsv, white_range[0], white_range[1])

    mask_wy = cv2.bitwise_or(mask_yellow, mask_white)
    h, w = mask_wy.shape

    region = np.array([[(0, h), (2*w//5, 4*h//7), (3*w//5, 4*h//7), (w, h)]])
    mask = np.zeros_like(mask_wy)
    mask = cv2.fillPoly(mask, region, 1)
    thresh = mask_wy * mask
    cv2.imshow('region', thresh)


if __name__ == '__main__':
    cap = cv2.VideoCapture('res/predict_turn.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('input', frame)
            thresh = removeBackNoise(frame)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
