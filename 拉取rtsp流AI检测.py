#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = 'geek_yolo26_det'
DEFAULT_RTSP_URL = 'rtsp://admin:sshw1234@192.168.5.145:554/cam/realmonitor?channel=1&subtype=0'
# DEFAULT_WEIGHT_FILE_NAME = 'det_person_helmet_250821.pt'
DEFAULT_WEIGHT_FILE_NAME = 'det_instrument_20260430.pt'
WINDOW_NAME = 'RTSP YOLO Real-time Detection'
LABEL_TRANSLATIONS = {
	'personup': '人',
	'nohelmet': '未戴安全帽',
	'helmet': '戴安全帽',
	'instrument': '仪表1',
 	'instrument_digital': '仪表2',
}
LABEL_COLORS = {
	'personup': (0, 200, 0),
	'helmet': (60, 170, 255),
	'nohelmet': (0, 0, 255),
	'instrument': (180, 90, 0),
	'instrument_digital': (90, 180, 0),
}
DEFAULT_WINDOW_MAX_WIDTH = 1280
DEFAULT_WINDOW_MAX_HEIGHT = 720


def is_display_available():
	return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def resolve_weights_path(weights):
	weights_path = Path(weights).expanduser()
	if weights_path.is_file():
		return weights_path.resolve()

	candidate_paths = [PACKAGE_DIR / weights_path.name]
	for parent in PACKAGE_DIR.parents:
		candidate_paths.append(parent / 'share' / PACKAGE_NAME / weights_path.name)
		candidate_paths.append(parent / weights_path.name)

	checked_paths = []
	seen_paths = set()
	for candidate in candidate_paths:
		candidate = candidate.resolve(strict=False)
		if candidate in seen_paths:
			continue
		seen_paths.add(candidate)
		checked_paths.append(candidate)
		if candidate.exists():
			return candidate

	checked = '\n'.join(str(path) for path in checked_paths)
	raise FileNotFoundError(f'YOLO weight file not found. Checked:\n{checked}')


def load_model(weights):
	weights_path = resolve_weights_path(weights)
	return YOLO(str(weights_path)), weights_path


def resolve_font_path():
	candidate_paths = [
		Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
		Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'),
		Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
		Path('/usr/share/fonts/truetype/arphic/ukai.ttc'),
	]

	for path in candidate_paths:
		if path.exists():
			return path

	raise FileNotFoundError('No Chinese font found. Please install Noto Sans CJK or WenQuanYi Zen Hei.')


def get_chinese_label(label):
	return LABEL_TRANSLATIONS.get(label, label)


def prepare_display_window(frame):
	frame_height, frame_width = frame.shape[:2]
	if frame_width <= 0 or frame_height <= 0:
		raise ValueError(f'Invalid frame size: {frame_width}x{frame_height}')

	window_scale = min(
		DEFAULT_WINDOW_MAX_WIDTH / frame_width,
		DEFAULT_WINDOW_MAX_HEIGHT / frame_height,
		1.0,
	)
	window_width = max(1, int(frame_width * window_scale))
	window_height = max(1, int(frame_height * window_scale))

	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
	cv2.resizeWindow(WINDOW_NAME, window_width, window_height)
	print(f'Camera frame size: {frame_width}x{frame_height}')
	print(f'Display window size: {window_width}x{window_height}')


def clamp_box_to_frame(box, frame_shape):
	frame_height, frame_width = frame_shape[:2]
	x1, y1, x2, y2 = [int(value) for value in box]
	x1 = max(0, min(x1, frame_width - 1))
	y1 = max(0, min(y1, frame_height - 1))
	x2 = max(0, min(x2, frame_width))
	y2 = max(0, min(y2, frame_height))
	if x2 <= x1 or y2 <= y1:
		return None
	return x1, y1, x2, y2


def print_available_labels(model):
	print('Available labels in model:')
	for class_id, label in model.names.items():
		print(f'  {class_id}: {label}')


def filter_results_by_labels(result, model, filter_labels):
	if not filter_labels:
		return result

	label_to_id = {label: class_id for class_id, label in model.names.items()}
	keep_class_ids = [label_to_id[label] for label in filter_labels if label in label_to_id]
	missing_labels = [label for label in filter_labels if label not in label_to_id]

	if missing_labels:
		print(f'Ignored unknown labels: {missing_labels}')

	if result.boxes is None:
		return result

	if not keep_class_ids:
		print('No filter labels matched the model classes. Falling back to all detections.')
		return result

	mask = None
	for class_id in keep_class_ids:
		class_mask = result.boxes.cls == class_id
		mask = class_mask if mask is None else (mask | class_mask)

	result.boxes = result.boxes[mask]
	return result


def draw_detections(frame, result, model, font, fps):
	pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	draw = ImageDraw.Draw(pil_image)

	detection_count = 0
	if result.boxes is not None:
		xyxy_list = result.boxes.xyxy.tolist()
		cls_list = result.boxes.cls.tolist()
		conf_list = result.boxes.conf.tolist()

		for box, class_id, confidence in zip(xyxy_list, cls_list, conf_list):
			clamped_box = clamp_box_to_frame(box, frame.shape)
			if clamped_box is None:
				continue

			detection_count += 1
			label = model.names[int(class_id)]
			translated_label = get_chinese_label(label)
			color = LABEL_COLORS.get(label, (0, 255, 0))
			x1, y1, x2, y2 = clamped_box

			draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

			text = f'{translated_label} {confidence:.2f}'
			text_bbox = draw.textbbox((0, 0), text, font=font)
			text_width = text_bbox[2] - text_bbox[0]
			text_height = text_bbox[3] - text_bbox[1]
			text_top = max(0, y1 - text_height - 8)
			draw.rectangle((x1, text_top, x1 + text_width + 12, text_top + text_height + 8), fill=color)
			draw.text((x1 + 6, text_top + 2), text, font=font, fill=(255, 255, 255))

	stats_text = f'FPS: {fps:.1f}   Detections: {detection_count}'
	stats_bbox = draw.textbbox((0, 0), stats_text, font=font)
	stats_width = stats_bbox[2] - stats_bbox[0]
	stats_height = stats_bbox[3] - stats_bbox[1]
	draw.rectangle((8, 8, 20 + stats_width, 20 + stats_height), fill=(0, 0, 0))
	draw.text((14, 12), stats_text, font=font, fill=(0, 255, 0))

	return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def open_rtsp_capture(rtsp_url, transport):
	if transport:
		os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'rtsp_transport;{transport}'

	cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
	return cap


def run_rtsp_detection(rtsp_url, conf, imgsz, weights, filter_labels, transport):
	display_enabled = is_display_available()
	if display_enabled:
		cv2.startWindowThread()
	else:
		print('DISPLAY/WAYLAND_DISPLAY is not set. Running inference without preview window.')

	model, weights_path = load_model(weights)
	font = ImageFont.truetype(str(resolve_font_path()), 24)
	cap = open_rtsp_capture(rtsp_url, transport)

	if not cap.isOpened():
		raise RuntimeError('Failed to open RTSP stream. Check URL, credentials, camera connectivity, and transport mode.')

	print(f'RTSP URL: {rtsp_url}')
	print(f'Weights: {weights_path}')
	print_available_labels(model)
	print(f'Filter labels: {filter_labels or "ALL"}')
	if display_enabled:
		print('Press q or Esc to exit')
	else:
		print('Press Ctrl+C to exit')

	frame_counter = 0
	fps = 0.0
	fps_window_start = time.perf_counter()
	window_prepared = False

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				print('Warning: failed to read frame, stream may have disconnected.')
				break

			if display_enabled and not window_prepared:
				prepare_display_window(frame)
				window_prepared = True

			results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
			filtered_result = filter_results_by_labels(results[0], model, filter_labels)

			frame_counter += 1
			current_time = time.perf_counter()
			elapsed_time = current_time - fps_window_start
			if elapsed_time >= 1.0:
				fps = frame_counter / elapsed_time
				fps_window_start = current_time
				frame_counter = 0

			annotated_frame = draw_detections(frame, filtered_result, model, font, fps)
			if display_enabled:
				cv2.imshow(WINDOW_NAME, annotated_frame)
				key = cv2.waitKey(1) & 0xFF
				if key == ord('q') or key == 27:
					break
	finally:
		cap.release()
		cv2.destroyAllWindows()


def parse_args():
	parser = argparse.ArgumentParser(description='Pull RTSP stream and run real-time YOLO detection.')
	parser.add_argument('--rtsp-url', default=DEFAULT_RTSP_URL, help='RTSP stream URL')
	parser.add_argument('--weights', default=DEFAULT_WEIGHT_FILE_NAME, help='YOLO .pt file path or file name')
	parser.add_argument('--labels', nargs='*', default=[], help='Labels to keep; omit to keep all classes')
	parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
	parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
	parser.add_argument('--rtsp-transport', choices=['tcp', 'udp'], default='tcp', help='RTSP transport mode for FFmpeg backend')
	return parser.parse_args()


def main():
	args = parse_args()
	run_rtsp_detection(args.rtsp_url, args.conf, args.imgsz, args.weights, args.labels, args.rtsp_transport)


if __name__ == '__main__':
	main()
