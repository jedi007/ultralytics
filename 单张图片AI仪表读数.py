from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from point_data_parser import DEFAULT_FRONT_VIEW_ANCHOR_POINTS, parse_readdata
from ultralytics import YOLO


# =========================
# Config (edit as needed)
# =========================
MODEL_PATH = "weights/det_instrument_20260817.pt"
IMAGE_PATH = "/data/清洗cache/识别误差/debug_havere_2026_08_26_17_39_18.jpg"  # 本地图片路径
CONF = 0.55
IOU = 0.45
IMGSZ = [384, 640]
DEVICE = None
WINDOW_NAME = "YOLO Detection Result"
QUIT_KEY = "q"

POSE_MODE_PATH = "weights/pose_instrument_m_260821_2.pt"
pose_cls_names = ["instrument"]
READING_KPT_CONF = 0.2


@dataclass
class PosePoint:
    x: int
    y: int
    conf: float = 1.0


@dataclass
class PoseDetection:
    cls_name: str
    box: tuple[int, int, int, int]
    points: list[PosePoint]
    reading: float | None = None


class PoseDetector:
    """Run pose estimation on cropped regions and map keypoints back to full frame."""

    def __init__(self, model_path: str | Path, target_names: list[str]) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"关键点模型文件不存在: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.target_names = set(target_names)

    @staticmethod
    def _clip_box(box, width, height):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(width, int(x2)), min(height, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def run_on_frame(self, frame, det_result, conf=0.25, iou=0.45):
        if det_result.boxes is None or len(det_result.boxes) == 0:
            return []

        h, w = frame.shape[:2]
        results = []

        for box_xyxy, cls_id in zip(det_result.boxes.xyxy.tolist(), det_result.boxes.cls.tolist()):
            cls_name = str(det_result.names.get(int(cls_id), cls_id))
            if cls_name not in self.target_names:
                continue

            clipped = self._clip_box(box_xyxy, w, h)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pose_results = self.model.predict(source=crop, conf=conf, iou=iou, verbose=False)
            if not pose_results or pose_results[0].keypoints is None:
                continue

            kpt = pose_results[0].keypoints
            if kpt.xy is None or len(kpt.xy) == 0:
                continue

            points = []
            xy = kpt.xy[0].tolist()
            confs = kpt.conf[0].tolist() if kpt.conf is not None else [1.0] * len(xy)
            for idx, (px, py) in enumerate(xy):
                points.append(PosePoint(x=int(px + x1), y=int(py + y1), conf=float(confs[idx])))

            reading = self._compute_reading(points)
            results.append(PoseDetection(cls_name=cls_name, box=(x1, y1, x2, y2), points=points, reading=reading))

        return results

    @staticmethod
    def _compute_reading(points):
        expected = len(DEFAULT_FRONT_VIEW_ANCHOR_POINTS) + 1
        if len(points) < expected:
            return None
        kpt_slice = points[:expected]
        if any(p.conf < READING_KPT_CONF for p in kpt_slice):
            return None
        try:
            reading, _, _, _ = parse_readdata([(p.x, p.y, p.conf) for p in kpt_slice])
            return float(reading)
        except Exception:
            return None

    @staticmethod
    def draw(frame, pose_detections, conf_thr=0.2):
        for det in pose_detections:
            for p in det.points:
                if p.conf >= conf_thr:
                    cv2.circle(frame, (p.x, p.y), 3, (0, 255, 255), -1)
            if det.reading is not None:
                x1, y1, _, _ = det.box
                cv2.putText(frame, f"reading: {det.reading:.2f}", (x1, max(30, y1 - 50)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 4, cv2.LINE_AA)


def main():
    det_model = YOLO(MODEL_PATH)
    pose_detector = PoseDetector(POSE_MODE_PATH, pose_cls_names)

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {IMAGE_PATH}")

    results = det_model.predict(source=frame, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE, verbose=False)
    result = results[0]

    annotated = result.plot(img=frame.copy())
    pose_dets = pose_detector.run_on_frame(frame, result, conf=CONF, iou=IOU)
    pose_detector.draw(annotated, pose_dets)

    # 打印检测结果
    box_count = 0 if result.boxes is None else len(result.boxes)
    print(f"检测到 {box_count} 个目标")
    for det in pose_dets:
        status = f"读数={det.reading:.2f}" if det.reading is not None else "读数失败"
        print(f"  {det.cls_name}: {status}")

    # 显示结果
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, annotated)
    print(f"按 '{QUIT_KEY}' 退出...")
    while True:
        if cv2.waitKey(100) & 0xFF == ord(QUIT_KEY):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
