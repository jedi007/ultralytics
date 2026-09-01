from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2

from USBCameraCapture import CameraCapture
from point_data_parser import DEFAULT_FRONT_VIEW_ANCHOR_POINTS, parse_readdata
from ultralytics import YOLO
from ultralytics.utils import TQDM


# =========================
# Config (edit as needed)
# =========================
MODEL_PATH = "weights/det_instrument_20260817.pt"
CAMERA_ID = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_FPS = 30
CONF = 0.55
IOU = 0.45
IMGSZ = [384, 640]
DEVICE = None

# 单目测距参数
FOCAL_LENGTH_MM = 5.4  # 摄像头焦距 (mm)
TARGET_SIZE_MM = 92.0    # 目标实际大小 (mm)
SENSOR_WIDTH_MM = 5.76   # 传感器宽度 (mm)，1/2.8英寸传感器

# 不同 label 对应的目标实际尺寸 (mm)
TARGET_SIZE_MM_BY_LABEL: dict[str, float] = {
    "instrument": 92.0,
    "instrument_led": 52.0,
}

# zoom_absolute=1 102px ==> 2.10m
F_DIV_SENSOR = 2.1558  # 这个是实测得出的，与设置的 SENSOR_WIDTH_MM 和 FOCAL_LENGTH_MM 无关

# zoom_absolute=100  95px ==> 2.65m
F_DIV_SENSOR = 2.5334

# zoom_absolute=200  99px ==> 2.77m
F_DIV_SENSOR = 2.7903

# zoom_absolute=300  116px ==> 2.77m
F_DIV_SENSOR = 3.2695

# zoom_absolute=400  135px ==> 2.77m
F_DIV_SENSOR = 3.8050

# # zoom_absolute=500  167px ==> 2.77m
# F_DIV_SENSOR = 4.6557

# # zoom_absolute=800  326px ==> 2.77m
# F_DIV_SENSOR = 9.1884


# zoom_absolute = 800
# F_DIV_SENSOR = 0.00530575 * zoom_absolute + 2.002825
# print(f"zoom_absolute={zoom_absolute} F_DIV_SENSOR={F_DIV_SENSOR:.4f}")

WINDOW_NAME = "YOLO Real-time Detection"
QUIT_KEY = "q"  # 按 q 退出
DISPLAY_SCALE = 1.0  # 显示窗口相对原图的缩放比例
MAX_DISPLAY_FPS = 50.0  # 显示帧率上限，避免播放过快


POSE_MODE_PATH = "weights/pose_instrument_m_260821_2.pt"
pose_cls_names = ["instrument", "instrument_led"]
READING_KPT_CONF = 0.2

min_value = 0
max_value = 2.5
total_value = max_value - min_value
readings = [min_value, min_value + total_value * 0.2, min_value + total_value * 0.4, min_value + total_value * 0.6, min_value + total_value * 0.8, max_value]

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
            reading, _, _, _ = parse_readdata(src_points, readings=readings)
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
        self.pose_detector = SecondaryPoseDetector(
            model_path=POSE_MODE_PATH,
            target_names=pose_cls_names,
        )

    def infer_stream(
        self,
        camera_id: int = 0,
        camera_width: int = 1920,
        camera_height: int = 1080,
        camera_fps: int = 30,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: list[int] = [384, 640],
        device: str | None = None,
        window_name: str = "YOLO Real-time Detection",
        quit_key: str = "q",
        display_scale: float = 1.0,
        max_display_fps: float = 15.0,
    ) -> InferenceStats:
        cam = CameraCapture(camera=camera_id, width=camera_width, height=camera_height, fps=camera_fps)
        try:
            cam.open()
        except RuntimeError as exc:
            raise RuntimeError(f"无法打开USB摄像头: {exc}") from exc

        stats = InferenceStats()

        progress = TQDM(
            total=None,
            desc="摄像头推理进度",
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
                success, frame = cam.read()
                if not success:
                    print("读取摄像头帧失败，退出。")
                    break

                frame_index += 1
                stats.total_frames += 1

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

                    # 单目测距：在检测框下方显示距离
                    if result.boxes is not None and len(result.boxes) > 0:
                        img_width = frame.shape[1]
                        img_height = frame.shape[0]

                        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                        cls_ids = result.boxes.cls.cpu().numpy()
                        for i, box in enumerate(boxes_xyxy):
                            x1, y1, x2, y2 = box
                            box_width_px = x2 - x1
                            box_height_px = y2 - y1
                            cls_id = int(cls_ids[i]) if i < len(cls_ids) else -1
                            cls_name = result.names.get(cls_id, str(cls_id))
                            print(f"[Frame {frame_index}] {cls_name} | w={box_width_px:.0f}px h={box_height_px:.0f}px")
                            # 取检测框的较大边作为目标尺寸（适用于不同朝向）
                            target_px = max(box_width_px, box_height_px)
                            target_size = TARGET_SIZE_MM_BY_LABEL.get(cls_name, TARGET_SIZE_MM)
                            
                            if target_px > 0:
                                distance_mm =  F_DIV_SENSOR * target_size * img_height / target_px
                                distance_m = distance_mm / 1000.0

                                # 在检测框下方显示距离
                                text = f"{distance_m:.2f}m"
                                text_x = int(x1)
                                text_y = int(y2) + 25
                                cv2.putText(
                                    annotated_frame,
                                    text,
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )

                    # 二级关键点检测：仅对指定类别进行扣图并回填关键点到原图
                    pose_detections = self.pose_detector.run_on_frame(
                        frame=frame,
                        det_result=result,
                        conf=conf,
                        iou=iou,
                    )
                    self.pose_detector.draw_keypoints(annotated_frame, pose_detections)

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

                key = cv2.waitKey(delay_ms) & 0xFF
                if key == key_code:
                    print(f"检测到退出按键: {quit_key}")
                    break
                elif key == ord("s"):
                    save_path = f"./测距存图/screenshot_{frame_index}.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"截图已保存: {save_path}")
                last_show_time = time.perf_counter()

        finally:
            progress.close()
            cam.release()
            cv2.destroyAllWindows()

        return stats


def main() -> None:
    detector = RealTimeVideoDetector(model_path=MODEL_PATH)

    stats = detector.infer_stream(
        camera_id=CAMERA_ID,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_fps=CAMERA_FPS,
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
    print(f"摄像头ID: {CAMERA_ID}")
    print("="*50)


if __name__ == "__main__":
    main()