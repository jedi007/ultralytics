from __future__ import annotations

import time

import cv2

from USBCameraCapture import CameraCapture


# =========================
# Config (edit as needed)
# =========================
CAMERA_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
WINDOW_NAME = "USB Camera"
QUIT_KEY = "q"  # 按 q 退出
DISPLAY_SCALE = 1.0  # 显示窗口相对原图的缩放比例
MAX_DISPLAY_FPS = 50.0  # 显示帧率上限，避免播放过快


def main() -> None:
    cam = CameraCapture(camera=CAMERA_ID, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    try:
        cam.open()
    except RuntimeError as exc:
        raise RuntimeError(f"无法打开USB摄像头: {exc}") from exc

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    key_code = ord(QUIT_KEY)
    window_size_inited = False
    min_frame_interval = 0.0 if MAX_DISPLAY_FPS <= 0 else 1.0 / MAX_DISPLAY_FPS
    last_show_time = 0.0

    try:
        while True:
            success, frame = cam.read()
            if not success:
                print("读取摄像头帧失败，退出。")
                break

            if DISPLAY_SCALE != 1.0:
                new_w = max(1, int(frame.shape[1] * DISPLAY_SCALE))
                new_h = max(1, int(frame.shape[0] * DISPLAY_SCALE))
                interp = cv2.INTER_CUBIC if DISPLAY_SCALE > 1.0 else cv2.INTER_AREA
                frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

            if not window_size_inited:
                cv2.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])
                window_size_inited = True

            cv2.imshow(WINDOW_NAME, frame)

            elapsed = time.perf_counter() - last_show_time
            remain = max(0.0, min_frame_interval - elapsed)
            delay_ms = max(1, int(remain * 1000)) if remain > 0 else 1

            if cv2.waitKey(delay_ms) & 0xFF == key_code:
                print(f"检测到退出按键: {QUIT_KEY}")
                break
            last_show_time = time.perf_counter()

    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
