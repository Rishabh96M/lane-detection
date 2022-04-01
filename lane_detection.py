# Lane Detection
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Lane detection used in early warning systems in self driving
#              cars

import cv2
import numpy as np

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
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        else:
            break
    cap.release()
    cv2.destroyAllWindows()
