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


class SecondaryPoseDetector:
    """Run pose estimation on cropped regions and map keypoints back to full frame."""

    def __init__(self, model_path: str | Path, target_names: list[str]) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"关键点模型文件不存在: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.target_names = set(target_names)

    @staticmethod
    def _clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box
        x1_i = max(0, min(int(x1), width - 1))
        y1_i = max(0, min(int(y1), height - 1))
        x2_i = max(0, min(int(x2), width))
        y2_i = max(0, min(int(y2), height))
        if x2_i <= x1_i or y2_i <= y1_i:
            return None
        return x1_i, y1_i, x2_i, y2_i

    @staticmethod
    def _resolve_name(names: dict[int, str] | list[str], cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    def run_on_frame(
        self,
        frame,
        det_result,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[PoseDetection]:
        if det_result.boxes is None or len(det_result.boxes) == 0:
            return []

        h, w = frame.shape[:2]
        mapped_results: list[PoseDetection] = []

        boxes_xyxy = det_result.boxes.xyxy.tolist()
        cls_ids = det_result.boxes.cls.tolist()

        for box_xyxy, cls_id in zip(boxes_xyxy, cls_ids):
            cls_name = self._resolve_name(det_result.names, int(cls_id))
            if cls_name not in self.target_names:
                continue

            clipped = self._clip_box(box_xyxy, w, h)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pose_results = self.model.predict(
                source=crop,
                conf=conf,
                iou=iou,
                verbose=False,
            )
            if not pose_results:
                continue
            pose_result = pose_results[0]
            if pose_result.keypoints is None or pose_result.keypoints.xy is None:
                continue
            if len(pose_result.keypoints.xy) == 0:
                continue

            # 取第一条姿态结果并映射回原图坐标
            pose_xy = pose_result.keypoints.xy[0].tolist()

            points: list[PosePoint] = []
            point_confs = None
            if pose_result.keypoints.conf is not None and len(pose_result.keypoints.conf) > 0:
                point_confs = pose_result.keypoints.conf[0].tolist()

            for idx, (px, py) in enumerate(pose_xy):
                conf_v = 1.0
                if point_confs is not None and idx < len(point_confs):
                    conf_v = float(point_confs[idx])
                points.append(PosePoint(x=int(px + x1), y=int(py + y1), conf=conf_v))

            reading = self._compute_reading(points)
            mapped_results.append(
                PoseDetection(
                    cls_name=cls_name,
                    box=(x1, y1, x2, y2),
                    points=points,
                    reading=reading,
                )
            )

        return mapped_results

    @staticmethod
    def _compute_reading(points: list[PosePoint]) -> float | None:
        expected_points = len(DEFAULT_FRONT_VIEW_ANCHOR_POINTS) + 1
        if len(points) < expected_points:
            return None

        kpt_slice = points[:expected_points]
        if any(point.conf < READING_KPT_CONF for point in kpt_slice):
            return None

        src_points = [(point.x, point.y, point.conf) for point in kpt_slice]
        try:
            reading, _, _, _ = parse_readdata(src_points)
            return float(reading)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def draw_keypoints(frame, pose_detections: list[PoseDetection], conf_thr: float = 0.2) -> None:
        for det in pose_detections:
            for p in det.points:
                if p.conf < conf_thr:
                    continue
                cv2.circle(frame, (p.x, p.y), 3, (0, 255, 255), -1)

            if det.reading is not None:
                x1, y1, _, _ = det.box
                text = f"reading: {det.reading:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(30, y1 - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA,
                )


def main() -> None:
    det_model = YOLO(MODEL_PATH)
    pose_detector = SecondaryPoseDetector(POSE_MODE_PATH, pose_cls_names)

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {IMAGE_PATH}")

    results = det_model.predict(source=frame, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE, verbose=False)
    result = results[0]

    annotated = result.plot(img=frame.copy())
    # 关键点检测使用较低的置信度阈值
    pose_dets = pose_detector.run_on_frame(frame, result, conf=READING_KPT_CONF, iou=IOU)
    pose_detector.draw_keypoints(annotated, pose_dets)

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
