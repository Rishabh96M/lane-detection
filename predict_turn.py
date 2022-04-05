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
    _, thresh = cv2.threshold(mask_wy, 200, 255, cv2.THRESH_BINARY)
    return thresh


def warp(image):
    dim = image.shape
    src = np.float32([[120, dim[0]-70], [dim[1]-120, dim[0]-70],
                     [4*dim[1]//7, 3*dim[0]//5], [3*dim[1]//7, 3*dim[0]//5]])
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


def detectLanes(image, windows=10, margin=100, thresh=50):
    histogram = np.sum(image[image.shape[0] // 3:, :], axis=0)
    output = np.dstack((image, image, image)) * 255

    leftx_base = np.argmax(histogram[:histogram.shape[0] // 2])
    rightx_base = np.argmax(
        histogram[histogram.shape[0] // 2:]) + histogram.shape[0] // 2

    window_height = np.int(image.shape[0] // windows)

    val_present = image.nonzero()
    val_present_y = np.array(val_present[0])
    val_present_x = np.array(val_present[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(windows):
        win_yb = image.shape[0] - (window + 1) * window_height
        win_yt = image.shape[0] - window * window_height
        win_xll = leftx_current - margin
        win_xlr = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrr = rightx_current + margin

        val_left = ((val_present_y >= win_yb) & (val_present_y < win_yt) & (
            val_present_x >= win_xll) & (val_present_x
                                         < win_xlr)).nonzero()[0]
        val_right = ((val_present_y >= win_yb) & (val_present_y < win_yt) & (
            val_present_x >= win_xrl) & (val_present_x
                                         < win_xrr)).nonzero()[0]

        left_lane_inds.append(val_left)
        right_lane_inds.append(val_right)

        if len(val_left) > thresh:
            leftx_current = np.int(np.mean(val_present_x[val_left]))
        if len(val_right) > thresh:
            rightx_current = np.int(np.mean(val_present_x[val_right]))

        cv2.rectangle(output, (win_xll, win_yb),
                      (win_xlr, win_yt), (0, 255, 0), 4)
        cv2.rectangle(output, (win_xrl, win_yb),
                      (win_xrr, win_yt), (0, 255, 0), 4)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = val_present_x[left_lane_inds]
    lefty = val_present_y[left_lane_inds]
    rightx = val_present_x[right_lane_inds]
    righty = val_present_y[right_lane_inds]

    return leftx, lefty, rightx, righty, output


def curveFitting(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    left_x = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_x = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left = np.array(np.vstack([left_x, ploty]).astype(np.int32).T)
    right = np.array(np.vstack([right_x, ploty]).astype(np.int32).T)

    y_max = np.max(ploty)
    l_rad_curv = ((1 + (2*left_fit[0]*y_max + left_fit[1])
                  ** 2)**1.5) / np.absolute(2*left_fit[0])
    r_rad_curv = ((1 + (2*right_fit[0]*y_max + right_fit[1])
                  ** 2)**1.5) / np.absolute(2*right_fit[0])

    return l_rad_curv, r_rad_curv, left, right


def colorLane(frame, left, right):
    input = frame.copy()
    warped, invH = warp(frame)
    cv2.polylines(warped, [left], False, [0, 0, 255], 5)
    cv2.polylines(warped, [right], False, [0, 0, 255], 5)
    points = np.append(left, np.flipud(right), 0)
    cv2.fillPoly(warped, np.int32([points]), (255, 0, 0))
    output = cv2.warpPerspective(
        warped, invH, (frame.shape[1], frame.shape[0]), frame)
    out = cv2.addWeighted(output, 0.8, input, 0.6, 1)
    return out


def pred_turn(l_rad_curv, r_rad_curv):
    turn = "STRAIGHT"
    rad_diff = l_rad_curv - r_rad_curv
    if(rad_diff > 800):
        turn = "RIGHT"
    if(rad_diff < -800):
        turn = "LEFT"
    return turn


def disp_output(turn, output, bin, warped, input, lanes_img, rad):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    cv2.putText(output, 'ACTION = ' + turn,
                (0, 50),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    cv2.putText(output, 'radius of curvature in pixels = '
                + str(rad),
                (0, 100),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    test = cv2.merge((warped, warped, warped))
    btm = cv2.resize(np.concatenate((test, lanes_img),
                     1), (750, 3*output.shape[0]//4))

    test2 = cv2.merge((bin, bin, bin))
    top = cv2.resize(np.concatenate((input, test2), 1),
                     (750, output.shape[0]//4))

    right = np.concatenate((top, btm), 0)

    return np.concatenate((output, right), 1)


if __name__ == '__main__':
    cap = cv2.VideoCapture('res/predict_turn.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            input = frame.copy()

            bin = removeBackNoise(frame)
            warped, invH = warp(bin)

            leftx, lefty, rightx, righty, lanes_img = detectLanes(warped)

            l_rad_curv, r_rad_curv, left_lane, right_lane = curveFitting(
                warped.shape, leftx, lefty, rightx, righty)

            turn = pred_turn(l_rad_curv, r_rad_curv)

            output = colorLane(frame, left_lane, right_lane)

            concat = disp_output(turn, output, bin, warped,
                                 input, lanes_img, (l_rad_curv + r_rad_curv)/2)

            cv2.imshow('output', concat)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
