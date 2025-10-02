import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_red_1 = np.array([0,150,150])
    upper_red_1 = np.array([10 ,255,255])

    lower_red_2 = np.array([170,150,150])
    upper_red_2 = np.array([179,255,255])

    mask1 = cv.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv.inRange(hsv, lower_red_2, upper_red_2)
    mask = mask1 + mask2

    open_close_mask = cv.erode(mask, kernel, iterations=1)
    open_close_mask = cv.dilate(open_close_mask, kernel, iterations=1)

    close_open_mask = cv.dilate(mask, kernel, iterations=1)
    close_open_mask = cv.erode(close_open_mask, kernel, iterations=1)

    result = cv.bitwise_and(frame, frame, mask=mask)
    open = cv.bitwise_and(frame, frame, mask=open_close_mask)
    close = cv.bitwise_and(frame, frame, mask=close_open_mask)

    # cv.imshow('Result', result)
    # cv.imshow('Open mask', open)
    # cv.imshow('Close mask', close)

    moments = cv.moments(mask)

    if moments["m00"] > 0:
        area = moments["m00"]

        cv.putText(frame, f"Area: {area:.0f} pixels", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)


    contour_y, contour_x = np.where(mask>0)

    if len(contour_x) > 0 and len(contour_y) > 0:
        x_min, x_max = np.min(contour_x), np.max(contour_x)
        y_min, y_max = np.min(contour_y), np.max(contour_y)

        cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv.imshow('Frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()