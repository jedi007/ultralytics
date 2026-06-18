from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


# =========================
# Config (edit as needed)
# =========================
MODEL_PATH = "det_smog_fire_250811.pt"
INPUT_DIR = "/data/CVAT标注/烟火数据集/det_smog_fire_250718_去重加上淘宝数据筛选/det_smog_fire/val/images"
OUTPUT_LABEL_DIR = "runs/det_smog_fire/pred_labels"
# None means auto: save to OUTPUT_LABEL_DIR parent directory as classes.txt
CLASSES_TXT_PATH = "runs/det_smog_fire/classes.txt"
CONF = 0.25
IOU = 0.45
IMGSZ = 640
DEVICE = None


@dataclass
class InferenceStats:
	total_images: int = 0
	success_images: int = 0
	failed_images: int = 0
	total_boxes: int = 0


class YoloLabelExporter:
	"""Batch inference utility that saves YOLO-format label files for each image."""

	IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

	def __init__(self, model_path: str | Path = "det_person_helmet_250821.pt") -> None:
		self.model_path = Path(model_path)
		if not self.model_path.exists():
			raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
		self.model = YOLO(str(self.model_path))

	def _iter_images(self, input_dir: str | Path) -> Iterable[Path]:
		input_path = Path(input_dir)
		if not input_path.exists() or not input_path.is_dir():
			raise NotADirectoryError(f"输入目录不存在或不是目录: {input_path}")

		return sorted(
			p
			for p in input_path.iterdir()
			if p.is_file() and p.suffix.lower() in self.IMAGE_SUFFIXES
		)

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
	def _write_yolo_label(label_file: Path, result) -> int:
		"""Write one image detection result to YOLO txt format and return box count."""
		label_file.parent.mkdir(parents=True, exist_ok=True)

		if result.boxes is None or len(result.boxes) == 0:
			label_file.write_text("", encoding="utf-8")
			return 0

		cls_list = result.boxes.cls.tolist()
		xywhn_list = result.boxes.xywhn.tolist()

		lines = []
		for cls_id, (x, y, w, h) in zip(cls_list, xywhn_list):
			lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

		label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
		return len(lines)

	def infer_folder(
		self,
		input_dir: str | Path,
		output_label_dir: str | Path,
		conf: float = 0.25,
		iou: float = 0.45,
		imgsz: int = 640,
		device: str | None = None,
	) -> InferenceStats:
		images = list(self._iter_images(input_dir))
		output_dir = Path(output_label_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

		stats = InferenceStats(total_images=len(images))
		if not images:
			return stats

		for image_path in images:
			label_path = output_dir / f"{image_path.stem}.txt"
			try:
				results = self.model.predict(
					source=str(image_path),
					conf=conf,
					iou=iou,
					imgsz=imgsz,
					device=device,
					verbose=False,
				)
				result = results[0]
				stats.total_boxes += self._write_yolo_label(label_path, result)
				stats.success_images += 1
			except Exception as exc:  # noqa: BLE001
				label_path.write_text("", encoding="utf-8")
				stats.failed_images += 1
				print(f"[失败] {image_path.name}: {exc}")

		return stats
def main() -> None:
	exporter = YoloLabelExporter(model_path=MODEL_PATH)
	classes_txt_path = Path(CLASSES_TXT_PATH) if CLASSES_TXT_PATH else Path(OUTPUT_LABEL_DIR).parent / "classes.txt"
	exporter.export_classes_txt(classes_txt_path)
	print(f"类别文件已保存: {classes_txt_path}")

	stats = exporter.infer_folder(
		input_dir=INPUT_DIR,
		output_label_dir=OUTPUT_LABEL_DIR,
		conf=CONF,
		iou=IOU,
		imgsz=IMGSZ,
		device=DEVICE,
	)

	print(f"总图片数: {stats.total_images}")
	print(f"成功处理: {stats.success_images}")
	print(f"失败数量: {stats.failed_images}")
	print(f"总框数量: {stats.total_boxes}")


if __name__ == "__main__":
	main()
