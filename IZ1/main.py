import cv2
import numpy as np
import time


def multi_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Ошибка: не удалось прочитать первый кадр.")
        return

    bbox = cv2.selectROI("Choose:", frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = map(int, bbox)
    roi = frame[y:y + h, x:x + w]
    cv2.destroyWindow("Choose:")

    # -------------------------------------
    # Настройка MeanShift
    # -------------------------------------
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))

    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    track_window_ms = (x, y, w, h)
    out_ms = cv2.VideoWriter("output-video/output_camshift.mp4",
                             cv2.VideoWriter_fourcc(*'mp4v'),
                             cap.get(cv2.CAP_PROP_FPS),
                             (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # -------------------------------------
    # Настройка CSRT
    # -------------------------------------
    tracker_csrt = cv2.TrackerCSRT_create()
    tracker_csrt.init(frame, bbox)
    out_csrt = cv2.VideoWriter("output-video/output_csrt.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               cap.get(cv2.CAP_PROP_FPS),
                               (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # -------------------------------------
    # Настройка KCF
    # -------------------------------------
    tracker_kcf = cv2.TrackerKCF_create()
    tracker_kcf.init(frame, bbox)
    out_kcf = cv2.VideoWriter("output-video/output_kcf.mp4",
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0
    start_time = time.time()

    fps_ms = 0
    fps_ms_sum = 0
    fps_csrt = 0
    fps_csrt_sum = 0
    fps_kcf = 0
    fps_kcf_sum = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # # -------------------------
        # # MeanShift
        # # -------------------------
        ms_start = time.time()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        _, track_window_ms = cv2.meanShift(dst, track_window_ms, term_crit)
        x_ms, y_ms, w_ms, h_ms = track_window_ms
        frame_ms = frame.copy()
        cv2.rectangle(frame_ms, (x_ms, y_ms), (x_ms + w_ms, y_ms + h_ms), (0, 255, 0), 2)

        ms_end = time.time()
        fps_ms = 1.0 / (ms_end - ms_start) if (ms_end - ms_start) > 0 else 0
        fps_ms_sum += fps_ms

        cv2.putText(frame_ms, f"MeanShift FPS: {fps_ms:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out_ms.write(frame_ms)
        cv2.imshow("CamShift", frame_ms)

        # -------------------------
        # CSRT
        # -------------------------
        csrt_start = time.time()
        success_c, bbox_c = tracker_csrt.update(frame)
        frame_csrt = frame.copy()
        if success_c:
            x_c, y_c, w_c, h_c = map(int, bbox_c)
            cv2.rectangle(frame_csrt, (x_c, y_c), (x_c + w_c, y_c + h_c), (255, 0, 0), 2)

        csrt_end = time.time()
        fps_csrt = 1.0 / (csrt_end - csrt_start) if (csrt_end - csrt_start) > 0 else 0
        fps_csrt_sum += fps_csrt

        cv2.putText(frame_csrt, f"CSRT FPS: {fps_csrt:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        out_csrt.write(frame_csrt)
        cv2.imshow("CSRT", frame_csrt)

        # -------------------------
        # KCF
        # -------------------------
        kcf_start = time.time()
        success_k, bbox_k = tracker_kcf.update(frame)
        frame_kcf = frame.copy()
        if success_k:
            x_k, y_k, w_k, h_k = map(int, bbox_k)
            cv2.rectangle(frame_kcf, (x_k, y_k), (x_k + w_k, y_k + h_k), (0, 0, 255), 2)

        kcf_end = time.time()
        fps_kcf = 1.0 / (kcf_end - kcf_start) if (kcf_end - kcf_start) > 0 else 0
        fps_kcf_sum += fps_kcf

        cv2.putText(frame_kcf, f"KCF FPS: {fps_kcf:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_kcf.write(frame_kcf)
        cv2.imshow("KCF", frame_kcf)

        current_time = time.time() - start_time
        overall_fps = frame_count / current_time if current_time > 0 else 0

        print(
            f"\rКадр: {frame_count} | MeanShift: {(fps_ms_sum / frame_count):.1f} FPS | CSRT: {fps_csrt:.1f} FPS | KCF: {fps_kcf:.1f} FPS | Общий: {overall_fps:.1f} FPS",
            end="", flush=True)

        if cv2.waitKey(10) & 0xFF == 27:
            break


    print(f"\n\n=== СТАТИСТИКА ===")
    print(f"Всего кадров: {frame_count}")
    print(f"Средний FPS (MeanShift): {(fps_ms_sum / frame_count):.1f}")
    print(f"Средний FPS (CSRT): {(fps_csrt_sum / frame_count):.1f}")
    print(f"Средний FPS (KCF): {(fps_kcf_sum / frame_count):.1f}")

    cap.release()
    out_ms.release()
    out_csrt.release()
    out_kcf.release()
    cv2.destroyAllWindows()


video_path = "video/birds.mp4"
multi_tracking(video_path)