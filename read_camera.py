import cv2 as cv

url = "http://192.168.1.85:4747/video"
cap = cv.VideoCapture(url)

if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    b, g, r = frame[center_x, center_y]
    if r >= g and r >= b:
        color = (0, 0, 255)
    elif g >= r and g >= b:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)

    vert_height = 100
    hor_width = 100
    bar_thickness = 20

    top_left_vert = (center_x - bar_thickness//2, center_y - vert_height//2)
    bottom_right_vert = (center_x + bar_thickness//2, center_y + vert_height//2)
    cv.rectangle(frame, top_left_vert, bottom_right_vert, color, -1)

    top_left_hor = (center_x - hor_width//2, center_y - bar_thickness//2)
    bottom_right_hor = (center_x + hor_width//2, center_y + bar_thickness//2)
    cv.rectangle(frame, top_left_hor, bottom_right_hor, color, -1)

    cv.imshow("Web-Camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv.destroyAllWindows()