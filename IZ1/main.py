import cv2

def multi_tracking(video_path):
    # Открываем видео
    cap = cv2.VideoCapture("boar1.mp4")
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    # Читаем первый кадр
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр.")
        return

    # Выбираем ROI (область интереса)
    bbox = cv2.selectROI("Choose:", frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = map(int, bbox)
    roi = frame[y:y + h, x:x + w]
    cv2.destroyWindow("Choose:")

    # =========================
    # Настройка MeanShift
    # =========================
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    track_window_ms = (x, y, w, h)
    out_ms = cv2.VideoWriter("IZ1/video/output_meanshift.avi",
                             cv2.VideoWriter_fourcc(*'XVID'),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # =========================
    # Настройка BOOSTING
    # =========================
    tracker_boosting = cv2.legacy.TrackerBoosting_create()
    tracker_boosting.init(frame, bbox)
    out_boosting = cv2.VideoWriter("IZ1/video/output_boosting.avi",
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   cap.get(cv2.CAP_PROP_FPS),
                                   (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # =========================
    # Настройка CSRT
    # =========================
    tracker_csrt = cv2.TrackerCSRT_create()
    tracker_csrt.init(frame, bbox)
    out_csrt = cv2.VideoWriter("IZ1/video/output_csrt.avi",
                               cv2.VideoWriter_fourcc(*'XVID'),
                               cap.get(cv2.CAP_PROP_FPS),
                               (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------
        # MeanShift
        # -------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        _, track_window_ms = cv2.meanShift(dst, track_window_ms, term_crit)
        x_ms, y_ms, w_ms, h_ms = track_window_ms
        frame_ms = frame.copy()
        cv2.rectangle(frame_ms, (x_ms, y_ms), (x_ms + w_ms, y_ms + h_ms), (0, 255, 0), 2)
        out_ms.write(frame_ms)
        cv2.imshow("MeanShift", frame_ms)

        # -------------------------
        # BOOSTING
        # -------------------------
        success_b, bbox_b = tracker_boosting.update(frame)
        frame_boost = frame.copy()
        if success_b:
            x_b, y_b, w_b, h_b = map(int, bbox_b)
            cv2.rectangle(frame_boost, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 0, 255), 2)
        out_boosting.write(frame_boost)
        cv2.imshow("BOOSTING", frame_boost)

        # -------------------------
        # CSRT
        # -------------------------
        success_c, bbox_c = tracker_csrt.update(frame)
        frame_csrt = frame.copy()
        if success_c:
            x_c, y_c, w_c, h_c = map(int, bbox_c)
            cv2.rectangle(frame_csrt, (x_c, y_c), (x_c + w_c, y_c + h_c), (255, 0, 0), 2)
        out_csrt.write(frame_csrt)
        cv2.imshow("CSRT", frame_csrt)

        # Выход по Esc
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # Закрываем все ресурсы
    cap.release()
    out_ms.release()
    out_boosting.release()
    out_csrt.release()
    cv2.destroyAllWindows()

video_path = "boar1.mp4"
multi_tracking(video_path)
