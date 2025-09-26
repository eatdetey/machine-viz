import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

ret, frame = cap.read()
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

lower_red_1 = np.array([0,150,150])
upper_red_1 = np.array([10 ,255,255])

lower_red_2 = np.array([170,150,150])
upper_red_2 = np.array([179,255,255])

mask1 = cv.inRange(hsv, lower_red_1, upper_red_1)
mask2 = cv.inRange(hsv, lower_red_2, upper_red_2)
mask = mask1 + mask2

result = cv.bitwise_and(frame, frame, mask=mask)

cv.imshow('Result', result)

cv.waitKey(0)
cv.destroyAllWindows()