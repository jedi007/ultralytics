from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from 云台测距采集数据.F_DIV_SENSOR_list import F_DIV_SENSOR_BY_ZOOM


# =========================
# 配置
# =========================
DEFAULT_MODEL_PATH = "weights/det_instrument_20260817.pt"
TARGET_SIZE_MM = 92.0
TARGET_SIZE_MM_BY_LABEL: dict[str, float] = {
    "instrument": 92.0,
    "instrument_led": 52.0,
}


@dataclass
class DetectionBox:
    """单个检测框信息"""
    x1: float
    y1: float
    x2: float
    y2: float
    cls_id: int
    cls_name: str
    confidence: float
    distance_m: float | None = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    """检测结果"""
    success: bool
    boxes: list[DetectionBox]
    abs_zoom: int | None = None
    f_div_sensor: float | None = None
    error: str | None = None
    yolo_result: Results | None = field(default=None, repr=False)


class DistanceDetector:
    """单目测距检测器

    输入 abs_zoom 和图像帧，返回带距离信息的检测框。
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        target_size_mm: float = TARGET_SIZE_MM,
        target_size_mm_by_label: dict[str, float] | None = None,
        conf: float = 0.55,
        iou: float = 0.45,
        imgsz: list[int] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        self.target_size_mm = target_size_mm
        self.target_size_mm_by_label = target_size_mm_by_label or TARGET_SIZE_MM_BY_LABEL
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz or [384, 640]

    def _get_f_div_sensor(self, abs_zoom: int) -> float | None:
        """根据 abs_zoom 获取 F_DIV_SENSOR"""
        if abs_zoom < 100:
            return None
        return F_DIV_SENSOR_BY_ZOOM.get(abs_zoom)

    def _calc_distance(
        self,
        f_div_sensor: float,
        target_px: float,
        target_size_mm: float,
        img_height: int,
    ) -> float | None:
        """计算距离 (米)"""
        if target_px <= 0:
            return None
        distance_mm = f_div_sensor * target_size_mm * img_height / target_px
        return distance_mm / 1000.0

    def detect(
        self,
        image: np.ndarray,
        abs_zoom: int,
    ) -> DetectionResult:
        """执行检测并计算距离

        Args:
            image: BGR 格式图像 (OpenCV 默认格式)
            abs_zoom: 当前焦距绝对值

        Returns:
            DetectionResult: 检测结果，包含 success 标志和 boxes 列表
        """
        if image is None or image.size == 0:
            return DetectionResult(success=False, boxes=[], error="图像为空")

        f_div_sensor = self._get_f_div_sensor(abs_zoom)
        img_height = image.shape[0]

        try:
            results = self.model.predict(
                source=image,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as e:
            return DetectionResult(success=False, boxes=[], abs_zoom=abs_zoom, error=str(e))

        if not results:
            return DetectionResult(success=False, boxes=[], abs_zoom=abs_zoom, error="无检测结果")

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionResult(success=True, boxes=[], abs_zoom=abs_zoom, f_div_sensor=f_div_sensor, yolo_result=result)

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        detection_boxes: list[DetectionBox] = []
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box
            cls_id = int(cls_ids[i]) if i < len(cls_ids) else -1
            cls_name = result.names.get(cls_id, str(cls_id))
            confidence = float(confs[i]) if i < len(confs) else 0.0

            # 计算距离
            distance_m = None
            if f_div_sensor is not None:
                target_px = max(x2 - x1, y2 - y1)
                target_size = self.target_size_mm_by_label.get(cls_name, self.target_size_mm)
                distance_m = self._calc_distance(f_div_sensor, target_px, target_size, img_height)

            detection_boxes.append(
                DetectionBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    cls_id=cls_id,
                    cls_name=cls_name,
                    confidence=confidence,
                    distance_m=distance_m,
                )
            )

        return DetectionResult(
            success=True,
            boxes=detection_boxes,
            abs_zoom=abs_zoom,
            f_div_sensor=f_div_sensor,
            yolo_result=result,
        )

    def draw(
        self,
        image: np.ndarray,
        result: DetectionResult,
        show_distance: bool = True,
        show_zoom: bool = True,
    ) -> np.ndarray:
        """在图像上绘制检测框和距离信息

        Args:
            image: 原始图像
            result: 检测结果
            show_distance: 是否显示距离
            show_zoom: 是否显示 abs_zoom

        Returns:
            绘制后的图像
        """
        annotated = image.copy()

        for box in result.boxes:
            # 绘制检测框
            x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 标签
            label = f"{box.cls_name} {box.conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 距离
            if show_distance and box.distance_m is not None:
                dist_text = f"{box.distance_m:.2f}m"
                cv2.putText(annotated, dist_text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 左上角显示 abs_zoom
        if show_zoom and result.abs_zoom is not None:
            zoom_text = f"abs_zoom={result.abs_zoom}"
            cv2.putText(annotated, zoom_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return annotated
