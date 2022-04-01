# Lane Detection
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Lane detection used in early warning systems in self driving
#              cars

import cv2
import numpy as np
import math

if __name__ == '__main__':
    cap = cv2.VideoCapture('res/straight_lane_detection.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow('input', frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            cv2.imshow('thesh', thresh)
            open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3)))
            cv2.imshow('open', open)

            edges = cv2.Canny(thresh, 100, 200, None, 3)
            cv2.imshow('edges', edges)

            lines = cv2.HoughLines(open, 1, np.pi / 180, 150, None, 0, 0)
            if lines is not None:
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow('out', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
