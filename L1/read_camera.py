import cv2 as cv

# url = "http://10.58.8.197:4747/video"
cap = cv.VideoCapture(0)

if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    right_bottom_x, right_bottom_y = 479, h-1

    b, g, r = frame[right_bottom_x, right_bottom_y]
    if r >= g and r >= b:
        color = (0, 0, 255)
    elif g >= r and g >= b:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    vert_height = 100
    hor_width = 100
    bar_thickness = 20

    bottom_1 = (center_x - 25, center_y - 25)
    top_1 = (center_x, center_y + 25)

    bottom_2 = (center_x + 25, center_y - 25)

    left_3 = (center_x - 25, center_y - 25)
    right_3 = (center_x + 25, center_y - 25)

    cv.line(frame, bottom_1, top_1, color, 5)
    cv.line(frame, bottom_2, top_1, color, 5)
    cv.line(frame, left_3, right_3, color, 5)

    bottom_4 = (center_x - 25, center_y + 25-15)
    top_4 = (center_x, center_y - 25-15)

    bottom_5 = (center_x + 25, center_y + 25-15)

    left_6 = (center_x - 25, center_y + 25-15)
    right_6 = (center_x + 25, center_y + 25-15)

    cv.line(frame, bottom_4, top_4, color, 5)
    cv.line(frame, bottom_5, top_4, color, 5)
    cv.line(frame, left_6, right_6, color, 5)

    # top_left_vert = (center_x - bar_thickness//2, center_y - vert_height//2)
    # bottom_right_vert = (center_x + bar_thickness//2, center_y + vert_height//2)
    # cv.rectangle(frame, top_left_vert, bottom_right_vert, color, -1)
    #
    # top_left_hor = (center_x - hor_width//2, center_y - bar_thickness//2)
    # bottom_right_hor = (center_x + hor_width//2, center_y + bar_thickness//2)
    # cv.rectangle(frame, top_left_hor, bottom_right_hor, color, -1)

    cv.imshow("Web-Camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv.destroyAllWindows()