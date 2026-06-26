from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from ultralytics import YOLO
from ultralytics.utils import TQDM


# =========================
# Config (edit as needed)
# =========================
MODEL_PATH = "det_smog_fire_250811.pt"
VIDEO_PATH = "/data/清洗cache/caiji/smoke_fire_1b1b6b2c69d959d5478c49e9bcdf8ad8.mp4"
OUTPUT_ROOT_DIR = "/data/清洗cache/caiji/video_pred"
OUTPUT_FRAME_DIR = f"{OUTPUT_ROOT_DIR}/frames"
OUTPUT_PRED_DIR = f"{OUTPUT_ROOT_DIR}/pred_frames"
OUTPUT_LABEL_DIR = f"{OUTPUT_ROOT_DIR}/labels"
# None means auto: save to OUTPUT_ROOT_DIR/classes.txt
CLASSES_TXT_PATH = None
CONF = 0.75
IOU = 0.45
IMGSZ = [384, 640]
DEVICE = None
FRAME_NAME_TEMPLATE = "frame_{frame_index:06d}"
SAVE_EMPTY_LABEL = True


@dataclass
class InferenceStats:
	total_frames: int = 0
	success_frames: int = 0
	failed_frames: int = 0
	total_boxes: int = 0


class VideoFrameLabelExporter:
	"""Run inference on each video frame and save images plus YOLO-format labels."""

	def __init__(self, model_path: str | Path = "det_person_helmet_250821.pt") -> None:
		self.model_path = Path(model_path)
		if not self.model_path.exists():
			raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
		self.model = YOLO(str(self.model_path))

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

	@staticmethod
	def _frame_stem(frame_index: int) -> str:
		return FRAME_NAME_TEMPLATE.format(frame_index=frame_index)

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
		try:
			while True:
				success, frame = cap.read()
				if not success:
					break

				frame_index += 1
				frame_stem = self._frame_stem(frame_index)
				frame_path = frame_dir / f"{frame_stem}.jpg"
				pred_path = pred_dir / f"{frame_stem}.jpg"
				label_path = label_dir / f"{frame_stem}.txt"

				try:
					if not cv2.imwrite(str(frame_path), frame):
						raise RuntimeError(f"原始帧保存失败: {frame_path}")

					results = self.model.predict(
						source=frame,
						conf=conf,
						iou=iou,
						imgsz=imgsz,
						device=device,
						verbose=False,
					)
					result = results[0]

					stats.total_boxes += self._write_yolo_label(label_path, result, save_empty_label=save_empty_label)

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
	exporter = VideoFrameLabelExporter(model_path=MODEL_PATH)
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
	)

	print(f"总帧数: {stats.total_frames}")
	print(f"成功处理: {stats.success_frames}")
	print(f"失败数量: {stats.failed_frames}")
	print(f"总框数量: {stats.total_boxes}")
	print(f"原始帧目录: {Path(OUTPUT_FRAME_DIR)}")
	print(f"推理结果图目录: {Path(OUTPUT_PRED_DIR)}")
	print(f"标签目录: {Path(OUTPUT_LABEL_DIR)}")


if __name__ == "__main__":
	main()
