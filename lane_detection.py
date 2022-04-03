# Lane Detection
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Lane detection used in early warning systems in self driving
#              cars

import cv2
import numpy as np


def make_points(image, average):
    _, slope, y_int = average
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])


if __name__ == '__main__':
    cap = cv2.VideoCapture('res/straight_lane_detection.mp4')
    preLin1 = []
    preLin2 = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            cv2.imshow('input', frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            mask = np.zeros_like(gray)
            region = np.array([[(100, h), (w // 2, 4 * h // 7), (w, h)]])
            mask = cv2.fillPoly(mask, region, 1)
            cv2.imshow('mask', gray * mask)
            _, thresh = cv2.threshold(gray * mask, 200, 255, cv2.THRESH_BINARY)
            cv2.imshow('thesh', thresh)

            lin1 = []
            lin2 = []

            linesP = cv2.HoughLinesP(
                thresh, 4, np.pi/180, 100, None, 10, 5)

            if linesP is not None:
                for i in range(0, len(linesP)):
                    pt = linesP[i][0]
                    m, y = np.polyfit((pt[0], pt[2]), (pt[1], pt[3]), 1)
                    d = abs(pt[3] - pt[1] + pt[2] - pt[0])
                    # cv2.line(frame, (pt[0], pt[1]),
                    #          (pt[2], pt[3]), [0, 0, 255], 3)
                    if m > 0:
                        lin1.append((d, m, y))
                    else:
                        lin2.append((d, m, y))
            preLin1 = np.average(lin1, 0)
            preLin2 = np.average(lin2, 0)
            if (preLin1[0] > preLin2[0]):
                pts = make_points(frame, preLin1)
                cv2.line(frame, (pts[0], pts[1]),
                         (pts[2], pts[3]), [0, 0, 255], 3)
                pts = make_points(frame, preLin2)
                cv2.line(frame, (pts[0], pts[1]),
                         (pts[2], pts[3]), [0, 255, 0], 3)
            else:
                pts = make_points(frame, preLin1)
                cv2.line(frame, (pts[0], pts[1]),
                         (pts[2], pts[3]), [0, 255, 0], 3)
                pts = make_points(frame, preLin2)
                cv2.line(frame, (pts[0], pts[1]),
                         (pts[2], pts[3]), [0, 0, 255], 3)

            cv2.imshow('out', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
