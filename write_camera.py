import cv2 as cv

def readWebWriteToFile():
    video = cv.VideoCapture(0)
    w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    video_writer = cv.VideoWriter("web-output.mov", fourcc, 30, (w,h))
    while True:
        ok, img = video.read()
        cv.imshow('img', img)
        video_writer.write(img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()

readWebWriteToFile()