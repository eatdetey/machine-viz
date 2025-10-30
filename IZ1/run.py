import cv2
from kcf import Tracker

if __name__ == '__main__':
    cap = cv2.VideoCapture(f"video/birds.mp4")
    out_kcf = cv2.VideoWriter("output-video/birds_own_kcf.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    tracker = Tracker()
    ok, frame = cap.read()
    if not ok:
        print("error reading video")
        exit(-1)
    roi = cv2.selectROI("tracking", frame, False, False)
    #roi = (218, 302, 148, 108)
    tracker.init(frame, roi)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        x, y, w, h = tracker.update(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        out_kcf.write(frame)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c==27 or c==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()