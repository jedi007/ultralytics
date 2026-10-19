from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from point_data_parser import DEFAULT_FRONT_VIEW_RADIUS, parse_readdata
from ultralytics import YOLO


# =========================
# Config (edit as needed)
# =========================
MODEL_PATH = "weights/det_instrument_20261013.pt"
IMAGE_PATH = "test_images/2026_10_11_15_42_32_debug_upload_frame_0_0.jpg"  # 本地图片路径
CONF = 0.55
IOU = 0.45
IMGSZ = [384, 640]
DEVICE = None
WINDOW_NAME = "YOLO Detection Result"
QUIT_KEY = "q"

POSE_MODE_PATH = "weights/pose_instrument_m_261016_mulit.pt"
pose_cls_names = ["instrument"]
READING_KPT_CONF = 0.2

# 各分类(cls_id)对应的刻度点读数，关键点顺序为: 表盘圆心, 指针针尖, 刻度点...
# cls_id: 0=8关键点(6个刻度点), 1=7关键点(5个刻度点), 2=9关键点(7个刻度点)
CLASS_SCALE_READINGS: dict[int, list[float]] = {
    0: [0, 20, 40, 60, 80, 100],
    1: [0, 25, 50, 75, 100],
    2: [0, 100 / 6, 100 / 3, 50, 200 / 3, 250 / 3, 100],
}
FRONT_VIEW_RADIUS = DEFAULT_FRONT_VIEW_RADIUS
FRONT_VIEW_START_ANGLE = 135.0
FRONT_VIEW_SWEEP_ANGLE = 270.0


def _build_front_view_anchors(num_scale_points: int) -> list[tuple[float, float]]:
    """按刻度点数量生成正视图锚点: index0=圆心, 其余为0读数到满量程均匀分布的刻度点。"""
    step = FRONT_VIEW_SWEEP_ANGLE / (num_scale_points - 1)
    anchors = [(FRONT_VIEW_RADIUS, FRONT_VIEW_RADIUS)]
    for i in range(num_scale_points):
        angle = math.radians(FRONT_VIEW_START_ANGLE + i * step)
        anchors.append(
            (
                FRONT_VIEW_RADIUS * (1.0 + math.cos(angle)),
                FRONT_VIEW_RADIUS * (1.0 + math.sin(angle)),
            )
        )
    return anchors


CLASS_FRONT_VIEW_ANCHORS: dict[int, list[tuple[float, float]]] = {
    cls_id: _build_front_view_anchors(len(readings)) for cls_id, readings in CLASS_SCALE_READINGS.items()
}


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

            # 姿态模型的分类决定仪表类型(即关键点解析方式)
            pose_cls_id = 0
            if pose_result.boxes is not None and pose_result.boxes.cls is not None and len(pose_result.boxes.cls) > 0:
                pose_cls_id = int(pose_result.boxes.cls[0].item())

            points: list[PosePoint] = []
            point_confs = None
            if pose_result.keypoints.conf is not None and len(pose_result.keypoints.conf) > 0:
                point_confs = pose_result.keypoints.conf[0].tolist()

            for idx, (px, py) in enumerate(pose_xy):
                conf_v = 1.0
                if point_confs is not None and idx < len(point_confs):
                    conf_v = float(point_confs[idx])
                points.append(PosePoint(x=int(px + x1), y=int(py + y1), conf=conf_v))

            reading = self._compute_reading(points, pose_cls_id)
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
    def _compute_reading(points: list[PosePoint], cls_id: int) -> float | None:
        scale_readings = CLASS_SCALE_READINGS.get(cls_id)
        if scale_readings is None:
            return None

        expected_points = len(scale_readings) + 2  # 圆心 + 指针针尖 + 各刻度点
        if len(points) < expected_points:
            return None

        kpt_slice = points[:expected_points]
        # if any(point.conf < READING_KPT_CONF for point in kpt_slice):
        #     return None

        src_points = [(point.x, point.y, point.conf) for point in kpt_slice]
        print(f"计算读数(cls_id={cls_id})，关键点坐标: {src_points}")
        try:
            reading, _, _, _ = parse_readdata(
                src_points,
                anchors=CLASS_FRONT_VIEW_ANCHORS[cls_id],
                readings=[scale_readings[0], *scale_readings],
            )
            return float(reading)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def draw_keypoints(frame, pose_detections: list[PoseDetection], conf_thr: float = 0.002) -> None:
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
