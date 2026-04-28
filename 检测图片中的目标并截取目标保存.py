#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import cv2
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / 'yolo26s_det_instrument_260427.pt'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# 脚本配置区：直接修改这些常量即可，无需命令行传参。
MODEL_PATH = DEFAULT_MODEL_PATH
INPUT_PATH = Path('/home/robot/gitlab/AIDATA/仪表/指针读数测试')
OUTPUT_PATH = Path('/home/robot/gitlab/AIDATA/仪表/指针读数测试_cropped')
CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = (384, 640)
CLASS_FILTER = None


def collect_images(input_path: Path):
	if input_path.is_file():
		if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
			raise ValueError(f'不支持的图片格式: {input_path}')
		return [input_path]

	if input_path.is_dir():
		image_paths = [
			path
			for path in sorted(input_path.rglob('*'))
			if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
		]
		if not image_paths:
			raise FileNotFoundError(f'目录下未找到图片: {input_path}')
		return image_paths

	raise FileNotFoundError(f'输入路径不存在: {input_path}')


def get_output_base_dir(image_path: Path, input_root: Path, output_root: Path):
	return output_root


def build_crop_name(image_path: Path, input_root: Path, index: int):
	if input_root.is_file():
		image_id = image_path.stem
	else:
		relative_stem = image_path.relative_to(input_root).with_suffix('')
		image_id = '__'.join(relative_stem.parts)

	return f'{image_id}_obj{index:03d}{image_path.suffix.lower()}'


def clamp_box(x1, y1, x2, y2, width, height):
	x1 = max(0, min(int(x1), width - 1))
	y1 = max(0, min(int(y1), height - 1))
	x2 = max(0, min(int(x2), width))
	y2 = max(0, min(int(y2), height))
	return x1, y1, x2, y2


def save_crops(result, image_path: Path, input_root: Path, output_root: Path, class_filter=None):
	boxes = result.boxes
	if boxes is None or len(boxes) == 0:
		return 0

	image = result.orig_img
	image_height, image_width = image.shape[:2]
	output_base_dir = get_output_base_dir(image_path, input_root, output_root)
	saved_count = 0

	for index, (xyxy, cls_id, conf) in enumerate(zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()), start=1):
		label = result.names[int(cls_id)]
		if class_filter and label not in class_filter:
			continue

		x1, y1, x2, y2 = clamp_box(*xyxy, image_width, image_height)
		if x2 <= x1 or y2 <= y1:
			continue

		crop = image[y1:y2, x1:x2]
		if crop.size == 0:
			continue

		crop_name = build_crop_name(image_path, input_root, index)
		crop_path = output_base_dir / crop_name

		if not cv2.imwrite(str(crop_path), crop):
			raise RuntimeError(f'保存裁剪图失败: {crop_path}')
		saved_count += 1

	return saved_count


def main():
	model_path = Path(MODEL_PATH).expanduser().resolve()
	input_path = Path(INPUT_PATH).expanduser().resolve()
	output_path = Path(OUTPUT_PATH).expanduser().resolve()
	class_filter = set(CLASS_FILTER) if CLASS_FILTER else None

	if not model_path.is_file():
		raise FileNotFoundError(f'模型文件不存在: {model_path}')

	image_paths = collect_images(input_path)
	output_path.mkdir(parents=True, exist_ok=True)

	model = YOLO(str(model_path))

	total_images = len(image_paths)
	total_crops = 0
	detected_images = 0

	print(f'模型文件: {model_path}')
	print(f'输入路径: {input_path}')
	print(f'输出路径: {output_path}')
	print(f'待处理图片数: {total_images}')
	if class_filter:
		print(f'类别过滤: {sorted(class_filter)}')

	for image_index, image_path in enumerate(image_paths, start=1):
		results = model.predict(source=str(image_path), conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE, verbose=False)
		saved_count = save_crops(results[0], image_path, input_path, output_path, class_filter)
		total_crops += saved_count
		if saved_count > 0:
			detected_images += 1
		print(f'[{image_index}/{total_images}] {image_path} -> 保存 {saved_count} 个目标')

	print(f'处理完成，共 {total_images} 张图片，检测到目标的图片 {detected_images} 张，保存裁剪图 {total_crops} 张。')


if __name__ == '__main__':
	main()
