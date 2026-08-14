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
VIDEO_PATH = "/data/清洗cache/video/26-08-14/rtsp_record_2026-08-14-10-12-07.mp4"
OUTPUT_ROOT_DIR = "/data/清洗cache/video/26-08-14/video_pred"
OUTPUT_FRAME_DIR = f"{OUTPUT_ROOT_DIR}/frames"
OUTPUT_PRED_DIR = f"{OUTPUT_ROOT_DIR}/pred_frames"
OUTPUT_LABEL_DIR = f"{OUTPUT_ROOT_DIR}/labels"
# None means auto: save to OUTPUT_ROOT_DIR/classes.txt
CLASSES_TXT_PATH = None
CONF = 0.55
IOU = 0.45
IMGSZ = [384, 640]
DEVICE = None
SAVE_EMPTY_LABEL = True

# 原有保存模式开关
# 1: 只有检测到目标才保存
# 2: 只有未检测到目标才保存
# 3: 全部帧都保存（原始逻辑）
SAVE_MODE = 2

# 新增画面相似度配置
FRAME_SIMILARITY_THRESHOLD = 0.95  # 相似度阈值，大于该值判定为画面重复跳过推理
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
    """Run inference on each video frame and save images plus YOLO-format labels."""

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
    def _write_yolo_label(label_file: Path, result, save_empty_label: bool = True) -> int:
        """Write one frame detection result to YOLO txt format and return box count."""
        label_file.parent.mkdir(parents=True, exist_ok=True)

        if result.boxes is None or len(result.boxes) == 0:
            if save_empty_label:
                label_file.write_text("", encoding="utf-8")
            return 0

        cls_list = result.boxes.cls.tolist()
        xywhn_list = result.boxes.xywhn.tolist()

        lines = []
        for cls_id, (x, y, w, h) in zip(cls_list, xywhn_list):
            lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return len(lines)

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
        output_frame_dir: str | Path,
        output_pred_dir: str | Path,
        output_label_dir: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: list[int] = [384, 640],
        device: str | None = None,
        save_empty_label: bool = True,
        save_mode: int = 3,
        sim_threshold: float = 0.85,
        sim_resize_w: int = 320,
        sim_resize_h: int = 180,
        hist_method=cv2.HISTCMP_CORREL,
    ) -> InferenceStats:
        video_file = Path(video_path)
        if not video_file.exists() or not video_file.is_file():
            raise FileNotFoundError(f"输入视频不存在: {video_file}")

        frame_dir = Path(output_frame_dir)
        pred_dir = Path(output_pred_dir)
        label_dir = Path(output_label_dir)
        for output_dir in (frame_dir, pred_dir, label_dir):
            output_dir.mkdir(parents=True, exist_ok=True)

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
                frame_path = frame_dir / f"{file_prefix}.bmp"
                pred_path = pred_dir / f"{file_prefix}.jpg"
                label_path = label_dir / f"{file_prefix}.txt"

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
                    has_object = result.boxes is not None and len(result.boxes) > 0

                    # 根据保存模式判断是否跳过文件保存
                    skip_save = False
                    if save_mode == 1 and not has_object:
                        skip_save = True
                    elif save_mode == 2 and has_object:
                        skip_save = True

                    if not skip_save:
                        # 保存原图
                        if not cv2.imwrite(str(frame_path), frame):
                            raise RuntimeError(f"原始帧保存失败: {frame_path}")
                        # 写入YOLO标签
                        box_cnt = self._write_yolo_label(label_path, result, save_empty_label=save_empty_label)
                        stats.total_boxes += box_cnt
                        # 保存绘制效果图
                        annotated_frame = result.plot()
                        if not cv2.imwrite(str(pred_path), annotated_frame):
                            raise RuntimeError(f"推理结果图保存失败: {pred_path}")

                    stats.success_frames += 1
                except Exception as exc:  # noqa: BLE001
                    if save_empty_label:
                        label_path.write_text("", encoding="utf-8")
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

    classes_txt_path = Path(CLASSES_TXT_PATH) if CLASSES_TXT_PATH else Path(OUTPUT_ROOT_DIR) / "classes.txt"
    exporter.export_classes_txt(classes_txt_path)
    print(f"类别文件已保存: {classes_txt_path}")

    stats = exporter.infer_video(
        video_path=VIDEO_PATH,
        output_frame_dir=OUTPUT_FRAME_DIR,
        output_pred_dir=OUTPUT_PRED_DIR,
        output_label_dir=OUTPUT_LABEL_DIR,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        save_empty_label=SAVE_EMPTY_LABEL,
        save_mode=SAVE_MODE,
        sim_threshold=FRAME_SIMILARITY_THRESHOLD,
        sim_resize_w=RESIZE_SIMILARITY_W,
        sim_resize_h=RESIZE_SIMILARITY_H,
        hist_method=HIST_COMPARE_METHOD
    )

    # 结果打印
    mode_desc = {1: "仅保存有目标帧", 2: "仅保存空帧", 3: "保存全部帧"}
    print("="*50)
    print(f"总帧数: {stats.total_frames}")
    print(f"画面相似跳过帧数: {stats.skip_similar_frames}")
    print(f"有效推理成功帧: {stats.success_frames}")
    print(f"推理失败帧: {stats.failed_frames}")
    print(f"检测目标总框数: {stats.total_boxes}")
    print(f"当前保存模式: {SAVE_MODE} - {mode_desc[SAVE_MODE]}")
    print(f"相似度判定阈值: {FRAME_SIMILARITY_THRESHOLD}")
    print(f"原始帧目录: {Path(OUTPUT_FRAME_DIR)}")
    print(f"推理结果图目录: {Path(OUTPUT_PRED_DIR)}")
    print(f"标签目录: {Path(OUTPUT_LABEL_DIR)}")
    print("="*50)


if __name__ == "__main__":
    main()