from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2

from ultralytics import YOLO
from ultralytics.utils import TQDM


# =========================
# Config (edit as needed)
# =========================
MODEL_PATH = "weights/det_instrument_20260806.pt"
VIDEO_PATH = "/data/清洗cache/已处理video/拍摄仪表/rtsp_record_2026-08-06-15-20-16.mp4"
CONF = 0.55
IOU = 0.45
IMGSZ = [384, 640]
DEVICE = None
WINDOW_NAME = "YOLO Real-time Detection"
QUIT_KEY = "q"  # 按 q 退出
DISPLAY_SCALE = 1.0  # 显示窗口相对原图的缩放比例
MAX_DISPLAY_FPS = 50.0  # 显示帧率上限，避免播放过快


@dataclass
class InferenceStats:
    total_frames: int = 0
    success_frames: int = 0
    failed_frames: int = 0
    total_boxes: int = 0


class RealTimeVideoDetector:
    """Run inference on a video stream and display annotated frames in real time."""

    def __init__(self, model_path: str | Path = "det_person_helmet_250821.pt") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self.model = YOLO(str(self.model_path))
    def infer_stream(
        self,
        video_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: list[int] = [384, 640],
        device: str | None = None,
        window_name: str = "YOLO Real-time Detection",
        quit_key: str = "q",
        display_scale: float = 1.0,
        max_display_fps: float = 15.0,
    ) -> InferenceStats:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频流: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats = InferenceStats(total_frames=max(total_frames, 0))

        progress = TQDM(
            total=total_frames if total_frames > 0 else None,
            desc="视频推理进度",
            unit="帧",
        )

        frame_index = 0
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        key_code = ord(quit_key)
        window_size_inited = False
        min_frame_interval = 0.0 if max_display_fps <= 0 else 1.0 / max_display_fps
        last_show_time = 0.0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_index += 1

                try:
                    results = self.model.predict(
                        source=frame,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                    )
                    result = results[0]
                    box_cnt = 0 if result.boxes is None else len(result.boxes)
                    stats.total_boxes += box_cnt

                    # 在原图上绘制检测框并显示，不写入任何 label 或图片文件
                    annotated_frame = result.plot(img=frame.copy())

                    if not window_size_inited:
                        show_w = max(1, int(frame.shape[1] * max(display_scale, 0.1)))
                        show_h = max(1, int(frame.shape[0] * max(display_scale, 0.1)))
                        cv2.resizeWindow(window_name, show_w, show_h)
                        window_size_inited = True

                    cv2.imshow(window_name, annotated_frame)

                    stats.success_frames += 1
                except Exception as exc:  # noqa: BLE001
                    stats.failed_frames += 1
                    print(f"[失败] 第 {frame_index} 帧: {exc}")
                finally:
                    progress.update(1)

                elapsed = time.perf_counter() - last_show_time
                remain = max(0.0, min_frame_interval - elapsed)
                delay_ms = max(1, int(remain * 1000)) if remain > 0 else 1

                if cv2.waitKey(delay_ms) & 0xFF == key_code:
                    print(f"检测到退出按键: {quit_key}")
                    break
                last_show_time = time.perf_counter()

            if stats.total_frames == 0:
                stats.total_frames = frame_index
        finally:
            progress.close()
            cap.release()
            cv2.destroyAllWindows()

        return stats


def main() -> None:
    detector = RealTimeVideoDetector(model_path=MODEL_PATH)

    stats = detector.infer_stream(
        video_path=VIDEO_PATH,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        window_name=WINDOW_NAME,
        quit_key=QUIT_KEY,
        display_scale=DISPLAY_SCALE,
        max_display_fps=MAX_DISPLAY_FPS,
    )

    # 结果打印
    print("="*50)
    print(f"总帧数: {stats.total_frames}")
    print(f"有效推理成功帧: {stats.success_frames}")
    print(f"推理失败帧: {stats.failed_frames}")
    print(f"检测目标总框数: {stats.total_boxes}")
    print(f"视频流地址: {VIDEO_PATH}")
    print("="*50)


if __name__ == "__main__":
    main()