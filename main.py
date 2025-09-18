import cv2 as cv

img1 = cv.imread("horse.jpg")
hsv = cv.cvtColor(img1, cv.COLOR_BGR2HSV)

cv.imshow('Horsen', img1)
cv.imshow('HSV Horsen', hsv)

# img2 = cv.imread("bigbob.jpeg", cv.IMREAD_COLOR_RGB)
# img3 = cv.imread("mefr.png", cv.IMREAD_REDUCED_COLOR_2)
#
# cv.namedWindow('Horsen window', cv.WINDOW_FREERATIO)
# cv.namedWindow('Big bob window', cv.WINDOW_KEEPRATIO)
# cv.namedWindow('Me and haters fr window', cv.WINDOW_AUTOSIZE)
#
# cv.imshow('Horsen window', img1)
# cv.imshow('Big bob window', img2)
# cv.imshow('Me and haters fr window', img3)
#
cv.waitKey(0)
cv.destroyAllWindows()

# cap = cv.VideoCapture("freaky-freak.mp4", cv.CAP_ANY)
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         new_size = cv.resize(frame, (600, 200))
#         cv.imshow("Video Freak", new_size)
#     else:
#         cap.set(cv.CAP_PROP_POS_FRAMES, 0)
#         continue
#     if cv.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv.destroyAllWindows()

# fps = cap.get(cv.CAP_PROP_FPS)
# width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# video_writer = cv.VideoWriter("copy_output.mov", fourcc, fps, (width,height))
# while (True):
#     ret, frame = cap.read()
#     cv.imshow('Video', frame)
#     video_writer.write(frame)
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv.destroyAllWindows()

