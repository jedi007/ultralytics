"""融合标注与推理标签。

功能：
1. 对比推理生成的 labels 与人工标注 labels。
2. 无争议样本不输出到结果目录。
3. 有争议样本输出融合 labels（人工 + 争议 AI）并复制对应原图。

默认按 YOLO 检测标签格式处理：
class x_center y_center width height [其他字段...]
仅使用前 5 列做 IoU 匹配，整行文本会原样写回输出文件。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

from ultralytics import YOLO


# =========================
# Config (可直接修改)
# =========================
MODEL_PATH = "det_smog_fire_250811.pt"
PRED_LABEL_DIR = "runs/det_smog_fire/pred_labels"
GT_LABEL_DIR = "/data/CVAT标注/烟火数据集/det_smog_fire_250718_去重加上淘宝数据筛选/det_smog_fire/val/labels"  # 例如数据集中的 labels 目录
SOURCE_IMAGE_DIR = "/data/CVAT标注/烟火数据集/det_smog_fire_250718_去重加上淘宝数据筛选/det_smog_fire/val/images"
OUTPUT_MERGED_DIR = "/data/清洗cache/yanhuomerge"
CLEAR_OUTPUT_SUBDIRS_BEFORE_RUN = True

# 仅当预测框与标注框类别相同且 IoU >= 阈值时，判定为一致
IOU_MATCH_THRESHOLD = 0.50


@dataclass
class YoloObject:
	cls_id: int
	x: float
	y: float
	w: float
	h: float
	source_index: int
	fields: list[str]
	raw_line: str


@dataclass
class MergeStats:
	total_files: int = 0
	files_with_disputes: int = 0
	files_without_disputes: int = 0
	gt_objects: int = 0
	pred_objects: int = 0
	disputed_pred_objects: int = 0
	output_label_files: int = 0
	copied_image_files: int = 0
	missing_image_files: int = 0


def load_label_file(label_path: Path) -> list[YoloObject]:
	"""读取单个 txt 标签文件，无法解析的行会被跳过。"""
	objects: list[YoloObject] = []
	if not label_path.exists():
		return objects

	for source_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
		raw = line.strip()
		if not raw:
			continue

		parts = raw.split()
		if len(parts) < 5:
			continue

		try:
			cls_id = int(float(parts[0]))
			x, y, w, h = map(float, parts[1:5])
		except ValueError:
			continue

		objects.append(
			YoloObject(
				cls_id=cls_id,
				x=x,
				y=y,
				w=w,
				h=h,
				source_index=source_index,
				fields=parts[1:],
				raw_line=raw,
			)
		)

	return objects


def xywh_to_xyxy(obj: YoloObject) -> tuple[float, float, float, float]:
	x1 = obj.x - obj.w / 2.0
	y1 = obj.y - obj.h / 2.0
	x2 = obj.x + obj.w / 2.0
	y2 = obj.y + obj.h / 2.0
	return x1, y1, x2, y2


def box_iou(a: YoloObject, b: YoloObject) -> float:
	ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
	bx1, by1, bx2, by2 = xywh_to_xyxy(b)

	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)

	inter_w = max(0.0, inter_x2 - inter_x1)
	inter_h = max(0.0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h

	area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = area_a + area_b - inter_area

	if union <= 0:
		return 0.0
	return inter_area / union


def load_model_class_names(model_path: Path) -> list[str]:
	"""从 pt 模型中读取类别名。"""
	model = YOLO(str(model_path))
	names = model.names
	if isinstance(names, dict):
		return [str(name) for _, name in sorted(names.items(), key=lambda x: x[0])]
	if isinstance(names, (list, tuple)):
		return [str(name) for name in names]
	raise TypeError(f"无法解析模型类别名称，names 类型: {type(names)}")


def build_output_class_names(base_class_names: list[str]) -> list[str]:
	"""构建输出类别列表：原始类别 + _human + _AI。"""
	human_class_names = [f"{name}_human" for name in base_class_names]
	ai_class_names = [f"{name}_AI" for name in base_class_names]
	return base_class_names + human_class_names + ai_class_names


def build_suffixed_id_maps(base_class_names: list[str]) -> tuple[dict[int, int], dict[int, int]]:
	"""构建原始 class_id 到 human/AI 新 class_id 的映射。"""
	base_count = len(base_class_names)
	human_id_map = {class_id: base_count + class_id for class_id in range(base_count)}
	ai_id_map = {class_id: base_count * 2 + class_id for class_id in range(base_count)}
	return human_id_map, ai_id_map


def format_with_new_class_id(obj: YoloObject, new_class_id: int) -> str:
	"""使用新的 class_id 重写 YOLO 标签行。"""
	return " ".join([str(new_class_id), *obj.fields])


def write_classes_txt(output_root_dir: Path, class_names: list[str]) -> Path:
	"""将输出类别名写入 classes.txt。"""
	classes_path = output_root_dir / "classes.txt"
	classes_path.parent.mkdir(parents=True, exist_ok=True)
	classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
	return classes_path


def evaluate_disputes_by_index(
	gt_objects: list[YoloObject],
	pred_objects: list[YoloObject],
	iou_threshold: float,
	) -> tuple[bool, list[int], list[int]]:
	"""基于类别一致且 IoU 达标的贪心匹配，返回剩余争议 GT/预测索引。"""
	gt_objects_copy = deepcopy(gt_objects)
	pred_objects_copy = deepcopy(pred_objects)

	for pred_obj in pred_objects_copy[:]:
		candidate_gt_objects = [
			gt_obj
			for gt_obj in gt_objects_copy
			if gt_obj.cls_id == pred_obj.cls_id and box_iou(gt_obj, pred_obj) >= iou_threshold
		]
		if not candidate_gt_objects:
			continue

		best_gt_obj = max(candidate_gt_objects, key=lambda gt_obj: box_iou(gt_obj, pred_obj))
		pred_objects_copy.remove(pred_obj)
		gt_objects_copy.remove(best_gt_obj)

	disputed_gt_indices = [obj.source_index for obj in gt_objects_copy]
	disputed_pred_indices = [obj.source_index for obj in pred_objects_copy]
	has_dispute = bool(disputed_gt_indices or disputed_pred_indices)
	return has_dispute, disputed_gt_indices, disputed_pred_indices


def build_merged_lines(
	gt_objects: list[YoloObject],
	pred_objects: list[YoloObject],
	disputed_gt_indices: list[int],
	disputed_pred_indices: list[int],
	human_id_map: dict[int, int],
	ai_id_map: dict[int, int],
) -> tuple[list[str], int]:
	"""仅对争议标签改写 class_id，其余标签保持人工标注原样。"""
	# disputed_gt_indices: 在贪心匹配后仍未被匹配消除的人工标签索引。
	disputed_gt_index_set = set(disputed_gt_indices)
	# disputed_pred_indices: 在贪心匹配后仍未被匹配消除的预测标签索引。
	disputed_pred_index_set = set(disputed_pred_indices)
	merged_lines: list[str] = []

	for gt_obj in gt_objects:
		if gt_obj.source_index in disputed_gt_index_set:
			merged_lines.append(format_with_new_class_id(gt_obj, human_id_map[gt_obj.cls_id]))
		else:
			merged_lines.append(gt_obj.raw_line)

	for pred_obj in pred_objects:
		if pred_obj.source_index in disputed_pred_index_set:
			merged_lines.append(format_with_new_class_id(pred_obj, ai_id_map[pred_obj.cls_id]))

	disputed_pred_count = len(disputed_pred_indices)
	return merged_lines, disputed_pred_count


def iter_label_stems(gt_dir: Path, pred_dir: Path) -> list[str]:
	gt_stems = {p.stem for p in gt_dir.glob("*.txt")}
	pred_stems = {p.stem for p in pred_dir.glob("*.txt")}
	return sorted(gt_stems | pred_stems)


def find_image_by_stem(image_dir: Path, stem: str) -> Path | None:
	"""按 stem 在图片目录中查找对应图片。"""
	for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
		candidate = image_dir / f"{stem}{suffix}"
		if candidate.exists():
			return candidate
	return None


def clear_files_in_dir(target_dir: Path) -> None:
	"""删除目录下的文件，避免历史结果残留。"""
	if not target_dir.exists():
		return
	for p in target_dir.iterdir():
		if p.is_file():
			p.unlink()


def merge_labels(
	gt_dir: Path,
	pred_dir: Path,
	source_image_dir: Path,
	output_root_dir: Path,
	iou_threshold: float,
	human_id_map: dict[int, int],
	ai_id_map: dict[int, int],
) -> MergeStats:
	labels_out_dir = output_root_dir / "labels"
	images_out_dir = output_root_dir / "images"
	labels_out_dir.mkdir(parents=True, exist_ok=True)
	images_out_dir.mkdir(parents=True, exist_ok=True)
	if CLEAR_OUTPUT_SUBDIRS_BEFORE_RUN:
		clear_files_in_dir(labels_out_dir)
		clear_files_in_dir(images_out_dir)

	stats = MergeStats()

	stems = iter_label_stems(gt_dir, pred_dir)
	stats.total_files = len(stems)

	for stem in stems:
		gt_path = gt_dir / f"{stem}.txt"
		pred_path = pred_dir / f"{stem}.txt"

		gt_objects = load_label_file(gt_path)
		pred_objects = load_label_file(pred_path)

		stats.gt_objects += len(gt_objects)
		stats.pred_objects += len(pred_objects)

		has_dispute, disputed_gt_indices, disputed_pred_indices = evaluate_disputes_by_index(
			gt_objects=gt_objects,
			pred_objects=pred_objects,
			iou_threshold=iou_threshold,
		)

		if has_dispute:
			stats.files_with_disputes += 1
			merged_lines, disputed_pred_count = build_merged_lines(
				gt_objects=gt_objects,
				pred_objects=pred_objects,
				disputed_gt_indices=disputed_gt_indices,
				disputed_pred_indices=disputed_pred_indices,
				human_id_map=human_id_map,
				ai_id_map=ai_id_map,
			)
			stats.disputed_pred_objects += disputed_pred_count

			# 仅争议样本输出融合标签
			out_label_path = labels_out_dir / f"{stem}.txt"
			text = "\n".join(merged_lines)
			if text:
				text += "\n"
			out_label_path.write_text(text, encoding="utf-8")
			stats.output_label_files += 1

			# 争议样本复制原始图片到输出 images 目录
			src_img = find_image_by_stem(source_image_dir, stem)
			if src_img is None:
				stats.missing_image_files += 1
			else:
				copy2(src_img, images_out_dir / src_img.name)
				stats.copied_image_files += 1
		else:
			stats.files_without_disputes += 1
			# 无争议时不输出任何文件

	return stats


def main() -> None:
	gt_dir = Path(GT_LABEL_DIR)
	pred_dir = Path(PRED_LABEL_DIR)
	source_image_dir = Path(SOURCE_IMAGE_DIR)
	output_root_dir = Path(OUTPUT_MERGED_DIR)
	model_path = Path(MODEL_PATH)

	if not gt_dir.exists() or not gt_dir.is_dir():
		raise NotADirectoryError(f"标注目录不存在或不是目录: {gt_dir}")
	if not pred_dir.exists() or not pred_dir.is_dir():
		raise NotADirectoryError(f"推理目录不存在或不是目录: {pred_dir}")
	if not source_image_dir.exists() or not source_image_dir.is_dir():
		raise NotADirectoryError(f"原始图片目录不存在或不是目录: {source_image_dir}")
	if not model_path.exists() or not model_path.is_file():
		raise FileNotFoundError(f"模型文件不存在: {model_path}")
	if not 0.0 <= IOU_MATCH_THRESHOLD <= 1.0:
		raise ValueError(f"IoU 阈值必须在 [0, 1] 内，当前: {IOU_MATCH_THRESHOLD}")

	base_class_names = load_model_class_names(model_path)
	output_class_names = build_output_class_names(base_class_names)
	human_id_map, ai_id_map = build_suffixed_id_maps(base_class_names)
	classes_txt_path = write_classes_txt(output_root_dir, output_class_names)

	stats = merge_labels(
		gt_dir=gt_dir,
		pred_dir=pred_dir,
		source_image_dir=source_image_dir,
		output_root_dir=output_root_dir,
		iou_threshold=IOU_MATCH_THRESHOLD,
		human_id_map=human_id_map,
		ai_id_map=ai_id_map,
	)

	print(f"已完成融合，输出目录: {output_root_dir}")
	print(f"类别映射文件: {classes_txt_path}")
	print(f"标签输出目录: {output_root_dir / 'labels'}")
	print(f"图片输出目录: {output_root_dir / 'images'}")
	print(f"处理文件数: {stats.total_files}")
	print(f"标注目标总数: {stats.gt_objects}")
	print(f"预测目标总数: {stats.pred_objects}")
	print(f"含争议文件数: {stats.files_with_disputes}")
	print(f"无争议文件数: {stats.files_without_disputes}")
	print(f"争议预测目标数: {stats.disputed_pred_objects}")
	print(f"输出融合标签文件数: {stats.output_label_files}")
	print(f"已复制图片数: {stats.copied_image_files}")
	print(f"缺失图片数: {stats.missing_image_files}")


if __name__ == "__main__":
	main()