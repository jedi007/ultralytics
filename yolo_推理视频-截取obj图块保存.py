from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils import TQDM


# =========================
# Config (edit as needed)
# =========================
MODEL_PATH = "weights/det_instrument_20260807.pt"
VIDEO_PATH = "/data/清洗cache/video/26-08-14/已处理/rtsp_record_2026-08-14-11-02-19.mp4"
OUTPUT_ROOT_DIR = "/data/清洗cache/video/26-08-14/video_pred6"
OUTPUT_OBJ_DIR = f"{OUTPUT_ROOT_DIR}/objects"

# 目标过滤配置
FILTER_LABEL_IDS = [0]  # None=不过滤, [0,1,2]=只保存这些类别
MIN_OBJ_SIZE = 256         # 最小目标面积(像素²), 0=不过滤
MAX_OBJ_SIZE = 0         # 最大目标面积(像素²), 0=不过滤

CONF = 0.35
IOU = 0.45
IMGSZ = [384, 640]
DEVICE = None

# 新增画面相似度配置
FRAME_SIMILARITY_THRESHOLD = 0.94  # 相似度阈值，大于该值判定为画面重复跳过推理
HIST_COMPARE_METHOD = cv2.HISTCMP_CORREL  # 直方图对比算法
RESIZE_SIMILARITY_W = 640
RESIZE_SIMILARITY_H = 384  # 缩小图做相似度计算，降低算力消耗


@dataclass
class InferenceStats:
    total_frames: int = 0
    skip_similar_frames: int = 0  # 新增：因画面相似跳过的帧数
    success_frames: int = 0
    failed_frames: int = 0
    total_boxes: int = 0


class VideoFrameLabelExporter:
    """Run inference on each video frame and save cropped object images."""

    def __init__(self, model_path: str | Path = "det_person_helmet_250821.pt", video_file_stem: str = "") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.video_stem = video_file_stem  # 视频文件名前缀
        # 缓存参考帧直方图，用于相似度对比
        self.ref_frame_hist = None

    def export_classes_txt(self, output_path: str | Path) -> Path:
        """Export model class names to classes.txt, one class name per line."""
        names = self.model.names
        if isinstance(names, dict):
            class_names = [str(name) for _, name in sorted(names.items(), key=lambda x: x[0])]
        elif isinstance(names, (list, tuple)):
            class_names = [str(name) for name in names]
        else:
            raise TypeError(f"无法解析模型类别名称，names 类型: {type(names)}")

        classes_path = Path(output_path)
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
        return classes_path

    def _get_file_prefix(self, frame_index: int) -> str:
        """生成文件名前缀：视频名_000001"""
        return f"{self.video_stem}_{frame_index:06d}"

    @staticmethod
    def calc_frame_hist(frame: np.ndarray, resize_w: int, resize_h: int) -> np.ndarray:
        """计算帧灰度直方图，用于相似度对比"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (resize_w, resize_h))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def infer_video(
        self,
        video_path: str | Path,
        output_obj_dir: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: list[int] = [384, 640],
        device: str | None = None,
        sim_threshold: float = 0.85,
        sim_resize_w: int = 320,
        sim_resize_h: int = 180,
        hist_method=cv2.HISTCMP_CORREL,
        filter_label_ids: list[int] | None = None,
        min_obj_size: int = 0,
        max_obj_size: int = 0,
    ) -> InferenceStats:
        video_file = Path(video_path)
        if not video_file.exists() or not video_file.is_file():
            raise FileNotFoundError(f"输入视频不存在: {video_file}")

        obj_dir = Path(output_obj_dir)
        obj_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_file}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stats = InferenceStats(total_frames=max(total_frames, 0))

        progress = TQDM(
            total=total_frames if total_frames > 0 else None,
            desc="视频推理进度",
            unit="帧",
        )

        frame_index = 0
        self.ref_frame_hist = None  # 重置参考帧直方图
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_index += 1
                file_prefix = self._get_file_prefix(frame_index)

                # 1. 画面相似度判断逻辑
                curr_hist = self.calc_frame_hist(frame, sim_resize_w, sim_resize_h)
                skip_by_similar = False
                if self.ref_frame_hist is not None:
                    # 计算直方图相似度
                    similarity = cv2.compareHist(self.ref_frame_hist, curr_hist, hist_method)
                    if similarity >= sim_threshold:
                        skip_by_similar = True
                        stats.skip_similar_frames += 1
                        progress.update(1)
                        continue
                # 画面变化明显，更新参考帧为当前帧
                self.ref_frame_hist = curr_hist

                # 2. 画面有变化，执行推理+保存逻辑
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

                    # 截取并保存检测到的目标
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        cls_list = boxes.cls.tolist()
                        xyxy_list = boxes.xyxy.tolist()
                        for obj_idx, (cls_id, (x1, y1, x2, y2)) in enumerate(zip(cls_list, xyxy_list)):
                            cls_id = int(cls_id)
                            # 类别过滤
                            if filter_label_ids is not None and cls_id not in filter_label_ids:
                                continue
                            # 尺寸过滤
                            obj_w = x2 - x1
                            obj_h = y2 - y1
                            obj_area = obj_w * obj_h
                            if min_obj_size > 0 and obj_area < min_obj_size:
                                continue
                            if max_obj_size > 0 and obj_area > max_obj_size:
                                continue
                            # 裁剪目标区域
                            crop_x1 = max(0, int(x1))
                            crop_y1 = max(0, int(y1))
                            crop_x2 = min(frame.shape[1], int(x2))
                            crop_y2 = min(frame.shape[0], int(y2))
                            cropped_obj = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            # 保存: 视频名_帧号_类别ID_crop_x1-crop_y1-crop_x2-crop_y2.jpg
                            obj_filename = f"{file_prefix}_{cls_id}_crop_{crop_x1}-{crop_y1}-{crop_x2}-{crop_y2}.jpg"
                            obj_path = obj_dir / obj_filename
                            if not cv2.imwrite(str(obj_path), cropped_obj):
                                print(f"[警告] 目标图块保存失败: {obj_path}")
                            stats.total_boxes += 1

                    stats.success_frames += 1
                except Exception as exc:  # noqa: BLE001
                    stats.failed_frames += 1
                    print(f"[失败] 第 {frame_index} 帧: {exc}")
                finally:
                    progress.update(1)

            if stats.total_frames == 0:
                stats.total_frames = frame_index
        finally:
            progress.close()
            cap.release()

        return stats


def main() -> None:
    # 提取视频文件名（不带后缀）
    video_path_obj = Path(VIDEO_PATH)
    video_name_stem = video_path_obj.stem
    exporter = VideoFrameLabelExporter(model_path=MODEL_PATH, video_file_stem=video_name_stem)

    stats = exporter.infer_video(
        video_path=VIDEO_PATH,
        output_obj_dir=OUTPUT_OBJ_DIR,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        sim_threshold=FRAME_SIMILARITY_THRESHOLD,
        sim_resize_w=RESIZE_SIMILARITY_W,
        sim_resize_h=RESIZE_SIMILARITY_H,
        hist_method=HIST_COMPARE_METHOD,
        filter_label_ids=FILTER_LABEL_IDS,
        min_obj_size=MIN_OBJ_SIZE,
        max_obj_size=MAX_OBJ_SIZE,
    )

    # 结果打印
    print("="*50)
    print(f"总帧数: {stats.total_frames}")
    print(f"画面相似跳过帧数: {stats.skip_similar_frames}")
    print(f"有效推理成功帧: {stats.success_frames}")
    print(f"推理失败帧: {stats.failed_frames}")
    print(f"检测目标总框数: {stats.total_boxes}")
    print(f"相似度判定阈值: {FRAME_SIMILARITY_THRESHOLD}")
    print(f"目标图块目录: {Path(OUTPUT_OBJ_DIR)}")
    print(f"类别过滤: {FILTER_LABEL_IDS}")
    print(f"尺寸过滤: min={MIN_OBJ_SIZE} max={MAX_OBJ_SIZE}")
    print("="*50)


if __name__ == "__main__":
    main()