# Predict Turn
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Predict the radius of curvature of the lane

import cv2
import numpy as np


def removeBackNoise(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_range = np.array(([[10, 100, 100], [45, 200, 255]]), np.uint8)
    mask_yellow = cv2.inRange(hsv, yellow_range[0], yellow_range[1])

    white_range = np.array(([[0, 0, 210], [255, 45, 255]]), np.uint8)
    mask_white = cv2.inRange(hsv, white_range[0], white_range[1])

    mask_wy = cv2.bitwise_or(mask_yellow, mask_white)
    return mask_wy


def warp(image):
    h, w = image.shape
    src = np.float32([[120, h-70], [w-120, h-70],
                     [4*w//7, 3*h//5], [3*w//7, 3*h//5]])
    dst = np.float32([[0, 1000], [1000, 1000], [1000, 0], [0, 0]])

    H = cv2.getPerspectiveTransform(src, dst)
    invH = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, H, (1000, 1000))
    return warped, invH


def checkForLine(seg, thresh=50):
    sum = np.sum(seg, 0)
    if sum > thresh:
        return seg.shape[0]//2, seg.shape[1]//2
    else:
        return -1


def detectLines(image, windows=10):
    hist = np.sum(image[image.shape[0]//4:, :], 0)
    leftx_pts = np.argmax(hist[:len(hist)//2])
    rightx_pts = np.argmax(hist[len(hist)//2:]) + len(hist)//2
    lefty_pts = image.shape[0] - 5
    righty_pts = image.shape[0] - 5
    out_img = np.dstack((image, image, image)) * 255

    window_size = image.shape[0] // windows
    margin = 100

    leftx_current = leftx_pts
    rightx_current = rightx_pts

    for window in range(windows):
        win_y_low = image.shape[0] - (window + 1) * window_size
        win_y_high = image.shape[0] - window * window_size
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 4)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 4)
    cv2.imshow('out', out_img)
    return leftx_pts, lefty_pts, rightx_pts, righty_pts


if __name__ == '__main__':
    cap = cv2.VideoCapture('res/predict_turn.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('input', frame)

            bin = removeBackNoise(frame)
            cv2.imshow('bin', bin)

            warped, invH = warp(bin)
            cv2.imshow('warped', warped)

            detectLines(warped)

            cv2.imshow('output', frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
