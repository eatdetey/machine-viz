import cv2 as cv

cap = cv.VideoCapture(0)

ret, frame = cap.read()
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
cv.imshow("HSV Frame", hsv)

cv.waitKey(0)
cv.destroyAllWindows()