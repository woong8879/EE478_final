#!/usr/bin/env python3
"""best.pt YOLO 성능 테스트 + 시각화.

사용 예:
  python3 yolo_test.py                       # D435i 컬러 스트림 라이브
  python3 yolo_test.py --source 0            # USB/웹캠 (v4l2 인덱스)
  python3 yolo_test.py --source img.jpg      # 단일 이미지 (창 표시 + 저장)
  python3 yolo_test.py --source folder/      # 폴더 일괄 (annotated/ 에 저장)
  python3 yolo_test.py --source clip.mp4     # 동영상

키: q 종료, s 현재 프레임 스냅샷 저장
"""
import argparse
import os
import time

import cv2
from ultralytics import YOLO


def draw_fps(frame, fps):
    cv2.putText(frame, f"{fps:5.1f} FPS", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def run_stream(get_frame, model, conf, release=None):
    n, t0, fps = 0, time.time(), 0.0
    os.makedirs("snaps", exist_ok=True)
    while True:
        frame = get_frame()
        if frame is None:
            break
        res = model.predict(frame, conf=conf, verbose=False)[0]
        vis = res.plot()                       # 박스+라벨+conf 그려진 BGR 이미지
        n += 1
        if n % 10 == 0:
            fps = 10.0 / (time.time() - t0)
            t0 = time.time()
        draw_fps(vis, fps)
        cv2.imshow("best.pt YOLO", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            fn = f"snaps/snap_{n}.jpg"
            cv2.imwrite(fn, vis)
            print("saved", fn)
    if release:
        release()
    cv2.destroyAllWindows()


def realsense_source():
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)

    def get():
        frames = pipe.wait_for_frames()
        c = frames.get_color_frame()
        if not c:
            return None
        import numpy as np
        return np.asanyarray(c.get_data())
    return get, pipe.stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--source", default="realsense",
                    help="realsense | 카메라인덱스(0) | 이미지/폴더/동영상 경로")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.model)
    print("classes:", model.names)

    src = args.source

    # 이미지/폴더/동영상 파일 → ultralytics 내장 처리(저장 포함)
    if src not in ("realsense",) and not src.isdigit():
        results = model.predict(src, conf=args.conf, save=True, verbose=True)
        # 단일 이미지면 창으로도 보여줌
        if len(results) == 1 and results[0].orig_img is not None:
            cv2.imshow("best.pt YOLO", results[0].plot())
            print("아무 키나 누르면 종료")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print("결과 저장 위치: runs/detect/predict*/")
        return

    if src == "realsense":
        try:
            get, release = realsense_source()
            run_stream(get, model, args.conf, release)
        except ImportError:
            print("pyrealsense2 없음 → D435i RGB(v4l2 /dev/video4)로 폴백")
            cap = cv2.VideoCapture(4)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            run_stream(lambda: (cap.read()[1] if cap.isOpened() else None),
                       model, args.conf, cap.release)
    else:
        cap = cv2.VideoCapture(int(src))
        run_stream(lambda: (cap.read()[1] if cap.isOpened() else None),
                   model, args.conf, cap.release)


if __name__ == "__main__":
    main()
