# Predict Turn
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Predict the radius of curvature of the lane

import cv2
import numpy as np


def removeBackNoise(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_range = np.array(([[5, 120, 80], [45, 200, 255]]), np.uint8)
    mask_yellow = cv2.inRange(hsv, yellow_range[0], yellow_range[1])

    white_range = np.array(([[0, 0, 210], [255, 45, 255]]), np.uint8)
    mask_white = cv2.inRange(hsv, white_range[0], white_range[1])

    mask_wy = cv2.bitwise_or(mask_yellow, mask_white)
    h, w = mask_wy.shape

    region = np.array
    return mask_wy


def warp(image):
    src = np.float32([[300, 670], [1090, 670], [740, 460], [580, 460]])
    dst = np.float32([[400, 0], [1200, 0], [300, 720], [1000, 720]])

    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(
        image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    cv2.imshow('warped', warped)


if __name__ == '__main__':
    cap = cv2.VideoCapture('res/predict_turn.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('input', frame)
            bin = removeBackNoise(frame)
            cv2.imshow('bin', bin)
            warp(bin)
            cv2.circle(frame, (300, 670), 4, [0, 0, 255], -1)
            cv2.circle(frame, (1090, 670), 4, [0, 0, 255], -1)
            cv2.circle(frame, (740, 460), 4, [0, 0, 255], -1)
            cv2.circle(frame, (580, 460), 4, [0, 0, 255], -1)
            cv2.imshow('output', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
