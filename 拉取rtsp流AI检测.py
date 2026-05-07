#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

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
DEFAULT_PTZ_PORT = 37777
DEFAULT_PTZ_CHANNEL = 0
DEFAULT_PTZ_SPEED = 8
DEFAULT_CENTER_DEADZONE = 100
DEFAULT_TARGET_LOST_TIMEOUT = 0.8
DEFAULT_PTZ_PULSE_DURATION = 0.5


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


def extract_rtsp_connection_info(rtsp_url):
	parsed = urlsplit(rtsp_url)
	return {
		'host': parsed.hostname,
		'username': parsed.username,
		'password': parsed.password,
	}


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


def select_tracking_target(result, model, tracking_labels):
	if result.boxes is None:
		return None

	best_target = None
	best_score = None
	for box, class_id, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
		label = model.names[int(class_id)]
		if tracking_labels and label not in tracking_labels:
			continue
		x1, y1, x2, y2 = [float(value) for value in box]
		width = max(0.0, x2 - x1)
		height = max(0.0, y2 - y1)
		area = width * height
		score = (area, float(confidence))
		if best_score is None or score > best_score:
			best_score = score
			best_target = {
				'label': label,
				'confidence': float(confidence),
				'box': (x1, y1, x2, y2),
			}

	return best_target


class PtzAutoTracker:
	def __init__(self, ptz_controller, deadzone_x, deadzone_y, target_lost_timeout):
		self.ptz_controller = ptz_controller
		self.deadzone_x = deadzone_x
		self.deadzone_y = deadzone_y
		self.target_lost_timeout = target_lost_timeout
		self.last_command_time = 0.0
		self.last_status = 'PTZ: idle'
		self.command_duration = 0.0
		self.command_thread = None
		self.command_error = None
		self.active_pulse_duration = None
		self.state_lock = threading.Lock()

	def update(self, frame_shape, target, speed, pulse_duration):
		now = time.perf_counter()
		self._update_command_completion_status()
		if target is None:
			self.last_status = 'PTZ: target lost'
			self.stop()
			return self.last_status

		if now - self.last_command_time < self.command_duration:
			return self.last_status
		self.last_command_time = now

		frame_height, frame_width = frame_shape[:2]
		x1, y1, x2, y2 = target['box']
		target_center_x = (x1 + x2) / 2.0
		target_center_y = (y1 + y2) / 2.0
		offset_x = target_center_x - frame_width / 2.0
		offset_y = target_center_y - frame_height / 2.0
  
		# print("self.deadzone_x: ", self.deadzone_x)
		# print("self.deadzone_y: ", self.deadzone_y)

		horizontal = None
		vertical = None
		if offset_x <= -self.deadzone_x:
			horizontal = 'left'
		elif offset_x >= self.deadzone_x:
			horizontal = 'right'

		if offset_y <= -self.deadzone_y:
			vertical = 'up'
		elif offset_y >= self.deadzone_y:
			vertical = 'down'

		if horizontal and vertical:
			next_command = f'{horizontal}{vertical}'
		elif horizontal:
			next_command = horizontal
		elif vertical:
			next_command = vertical
		else:
			next_command = None

		if next_command is None:
			self.stop()
			self.last_status = (
				f"PTZ: centered dx={offset_x:.3f} dy={offset_y:.3f}, next_command is None"
			)
			return self.last_status

		x_duration = abs(offset_x) / 300.0 if abs(offset_x) > self.deadzone_x else 5.0
		y_duration = abs(offset_y) / 300.0 if abs(offset_y) > self.deadzone_y else 5.0
		pulse_duration = max(min(x_duration, y_duration), 0.1)
		self.command_duration = pulse_duration
		self._start_command_pulse(next_command, speed, pulse_duration)
		self.last_status = f'PTZ: pulse {next_command} speed={speed} duration={pulse_duration:.2f}s dx={offset_x:.3f} dy={offset_y:.3f}'
		return self.last_status

	def _start_command_pulse(self, command_name, speed, pulse_duration):
		with self.state_lock:
			self.active_pulse_duration = pulse_duration
			self.command_error = None
			self.command_thread = threading.Thread(
				target=self._run_command_pulse,
				args=(command_name, speed, pulse_duration),
				name='ptz-command-pulse',
				daemon=True,
			)
			self.command_thread.start()

	def _run_command_pulse(self, command_name, speed, pulse_duration):
		try:
			self.ptz_controller.ptz_control(command_name, speed, pulse_duration)
		except Exception as exc:
			with self.state_lock:
				self.command_error = exc
		finally:
			with self.state_lock:
				self.active_pulse_duration = None
				self.command_thread = None
				self.last_command_time = time.perf_counter()

	def _update_command_completion_status(self):
		with self.state_lock:
			command_error = self.command_error
			self.command_error = None

		if command_error is not None:
			self.last_status = f'PTZ: command error {command_error}'
		elif self.last_status.startswith('PTZ: pulse '):
			self.last_status = 'PTZ: idle'

	def stop(self, wait=False):
		with self.state_lock:
			command_thread = self.command_thread
			active_pulse_duration = self.active_pulse_duration

		if wait and command_thread is not None:
			join_timeout = (active_pulse_duration or DEFAULT_PTZ_PULSE_DURATION) + 1.0
			command_thread.join(timeout=join_timeout)


def create_ptz_controller(rtsp_url, host, port, username, password, channel):
	try:
		from dahua_control_demo import DahuaPtzDemo
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError('NetSDK is required for --auto-center. Ensure dahua_control_demo.py dependencies are installed.') from exc

	connection_info = extract_rtsp_connection_info(rtsp_url)
	resolved_host = host or connection_info['host']
	resolved_username = username or connection_info['username'] or 'admin'
	resolved_password = password or connection_info['password'] or ''
	if not resolved_host:
		raise ValueError('PTZ host is required. Provide --ptz-host or embed the host in --rtsp-url.')

	ptz_controller = DahuaPtzDemo(
		host=resolved_host,
		port=port,
		username=resolved_username,
		password=resolved_password,
		channel=channel,
	)
	ptz_controller.login()
	print(f'PTZ auto-center enabled: {resolved_host}:{port}, channel={channel}')
	return ptz_controller


def draw_detections(frame, result, model, font, fps, tracking_target=None, tracking_status=None):
	pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	draw = ImageDraw.Draw(pil_image)
	frame_height, frame_width = frame.shape[:2]
	center_x = frame_width // 2
	center_y = frame_height // 2
	draw.line((center_x - 20, center_y, center_x + 20, center_y), fill=(255, 255, 0), width=2)
	draw.line((center_x, center_y - 20, center_x, center_y + 20), fill=(255, 255, 0), width=2)

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

	if tracking_target is not None:
		x1, y1, x2, y2 = [int(value) for value in tracking_target['box']]
		target_center_x = (x1 + x2) // 2
		target_center_y = (y1 + y2) // 2
		draw.ellipse((target_center_x - 6, target_center_y - 6, target_center_x + 6, target_center_y + 6), fill=(255, 255, 0))
		draw.line((center_x, center_y, target_center_x, target_center_y), fill=(255, 255, 0), width=2)

	stats_text = f'FPS: {fps:.1f}   Detections: {detection_count}'
	if tracking_status:
		stats_text = f'{stats_text}   {tracking_status}'
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
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
	cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
	return cap


class LatestFrameCapture:
	def __init__(self, rtsp_url, transport):
		self.cap = open_rtsp_capture(rtsp_url, transport)
		self.lock = threading.Lock()
		self.frame = None
		self.frame_id = 0
		self.running = False
		self.thread = None
		self.read_failed = False

	def start(self):
		if not self.cap.isOpened():
			return self
		self.running = True
		self.thread = threading.Thread(target=self._reader_loop, name='rtsp-latest-frame', daemon=True)
		self.thread.start()
		return self

	def _reader_loop(self):
		while self.running:
			ret, frame = self.cap.read()
			if not ret:
				self.read_failed = True
				self.running = False
				break

			with self.lock:
				self.frame = frame
				self.frame_id += 1

	def read(self, last_frame_id=None, timeout=1.0):
		deadline = time.perf_counter() + timeout
		while time.perf_counter() < deadline:
			with self.lock:
				if self.frame is not None and self.frame_id != last_frame_id:
					return True, self.frame.copy(), self.frame_id

			if not self.running:
				break

			time.sleep(0.001)

		with self.lock:
			if self.frame is not None:
				return True, self.frame.copy(), self.frame_id

		return False, None, last_frame_id

	def release(self):
		self.running = False
		if self.thread is not None and self.thread.is_alive():
			self.thread.join(timeout=1.0)
		self.cap.release()


def run_rtsp_detection(
	rtsp_url,
	conf,
	imgsz,
	weights,
	filter_labels,
	transport,
	auto_center,
	center_target_labels,
	ptz_host,
	ptz_port,
	ptz_username,
	ptz_password,
	ptz_channel,
	ptz_speed,
	ptz_pulse_duration,
	center_deadzone_x,
	center_deadzone_y,
	target_lost_timeout,
):
	display_enabled = is_display_available()
	if display_enabled:
		cv2.startWindowThread()
	else:
		print('DISPLAY/WAYLAND_DISPLAY is not set. Running inference without preview window.')

	model, weights_path = load_model(weights)
	font = ImageFont.truetype(str(resolve_font_path()), 24)
	capture = LatestFrameCapture(rtsp_url, transport).start()
	ptz_controller = None
	auto_tracker = None

	if not capture.cap.isOpened():
		raise RuntimeError('Failed to open RTSP stream. Check URL, credentials, camera connectivity, and transport mode.')
	if auto_center:
		ptz_controller = create_ptz_controller(rtsp_url, ptz_host, ptz_port, ptz_username, ptz_password, ptz_channel)
		auto_tracker = PtzAutoTracker(
			ptz_controller=ptz_controller,
			deadzone_x=center_deadzone_x,
			deadzone_y=center_deadzone_y,
			target_lost_timeout=target_lost_timeout,
		)

	print(f'RTSP URL: {rtsp_url}')
	print(f'Weights: {weights_path}')
	print_available_labels(model)
	print(f'Filter labels: {filter_labels or "ALL"}')
	if auto_center:
		print(f'Center target labels: {center_target_labels or filter_labels or "ALL"}')
	if display_enabled:
		print('Press q or Esc to exit')
	else:
		print('Press Ctrl+C to exit')

	frame_counter = 0
	fps = 0.0
	fps_window_start = time.perf_counter()
	window_prepared = False
	last_frame_id = None
	last_status = None

	try:
		while True:
			ret, frame, last_frame_id = capture.read(last_frame_id=last_frame_id, timeout=2.0)
			if not ret:
				print('Warning: failed to read frame, stream may have disconnected.')
				break

			if display_enabled and not window_prepared:
				prepare_display_window(frame)
				window_prepared = True

			results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
			filtered_result = filter_results_by_labels(results[0], model, filter_labels)
			tracking_labels = center_target_labels or filter_labels
			tracking_target = select_tracking_target(filtered_result, model, tracking_labels)
			# print("tracking_target: ", tracking_target)
			# print("frame.shape: ", frame.shape)
			#   tracking_target:  {'label': 'instrument', 'confidence': 0.3569718897342682, 'box': (824.5673828125, 425.1272277832031, 942.43310546875, 541.5030517578125)}
			#   frame.shape:  (1080, 1920, 3)
			tracking_status = (
				auto_tracker.update(frame.shape, tracking_target, ptz_speed, ptz_pulse_duration)
				if auto_tracker else None
			)
			if tracking_status != last_status:
				print("tracking_status: ", tracking_status)
				last_status = tracking_status

			frame_counter += 1
			current_time = time.perf_counter()
			elapsed_time = current_time - fps_window_start
			if elapsed_time >= 1.0:
				fps = frame_counter / elapsed_time
				fps_window_start = current_time
				frame_counter = 0

			annotated_frame = draw_detections(
				frame,
				filtered_result,
				model,
				font,
				fps,
				tracking_target=tracking_target,
				tracking_status=tracking_status,
			)
			if display_enabled:
				cv2.imshow(WINDOW_NAME, annotated_frame)
				key = cv2.waitKey(1) & 0xFF
				if key == ord('q') or key == 27:
					break
	finally:
		if auto_tracker is not None:
			auto_tracker.stop(wait=True)
		if ptz_controller is not None:
			ptz_controller.stop_active_ptz()
			ptz_controller.cleanup()
		capture.release()
		cv2.destroyAllWindows()


def parse_args():
	parser = argparse.ArgumentParser(description='Pull RTSP stream and run real-time YOLO detection.')
	parser.add_argument('--rtsp-url', default=DEFAULT_RTSP_URL, help='RTSP stream URL')
	parser.add_argument('--weights', default=DEFAULT_WEIGHT_FILE_NAME, help='YOLO .pt file path or file name')
	parser.add_argument('--labels', nargs='*', default=[], help='Labels to keep; omit to keep all classes')
	parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
	parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
	parser.add_argument('--rtsp-transport', choices=['tcp', 'udp'], default='tcp', help='RTSP transport mode for FFmpeg backend')
	parser.add_argument('--auto-center', default=True, action='store_true', help='Enable Dahua PTZ auto-centering for the selected target')
	parser.add_argument('--center-target-labels', nargs='*', default=[], help='Labels eligible for PTZ auto-centering; omit to reuse --labels or all detections')
	parser.add_argument('--ptz-host', default=None, help='Dahua PTZ device IP, defaults to the host parsed from --rtsp-url')
	parser.add_argument('--ptz-port', type=int, default=DEFAULT_PTZ_PORT, help='Dahua NetSDK port')
	parser.add_argument('--ptz-username', default=None, help='Dahua NetSDK username, defaults to the username parsed from --rtsp-url')
	parser.add_argument('--ptz-password', default=None, help='Dahua NetSDK password, defaults to the password parsed from --rtsp-url')
	parser.add_argument('--ptz-channel', type=int, default=DEFAULT_PTZ_CHANNEL, help='Dahua PTZ channel number')
	parser.add_argument('--ptz-speed', type=int, default=DEFAULT_PTZ_SPEED, help='PTZ speed, range 1-8')
	parser.add_argument('--ptz-pulse-duration', type=float, default=DEFAULT_PTZ_PULSE_DURATION, help='Pulse duration in seconds for each PTZ command')
	parser.add_argument('--center-deadzone-x', type=float, default=DEFAULT_CENTER_DEADZONE, help='Horizontal deadzone ratio around image center')
	parser.add_argument('--center-deadzone-y', type=float, default=DEFAULT_CENTER_DEADZONE, help='Vertical deadzone ratio around image center')
	parser.add_argument('--target-lost-timeout', type=float, default=DEFAULT_TARGET_LOST_TIMEOUT, help='Seconds to wait before stopping PTZ after target loss')
	return parser.parse_args()


def main():
	args = parse_args()
	print("args.auto_center: ", args.auto_center)
	print("args.ptz_speed", args.ptz_speed)
	print("args.center_deadzone_x", args.center_deadzone_x)
	print("args.center_deadzone_y", args.center_deadzone_y)
	run_rtsp_detection(
		args.rtsp_url,
		args.conf,
		args.imgsz,
		args.weights,
		args.labels,
		args.rtsp_transport,
		args.auto_center,
		args.center_target_labels,
		args.ptz_host,
		args.ptz_port,
		args.ptz_username,
		args.ptz_password,
		args.ptz_channel,
		args.ptz_speed,
		args.ptz_pulse_duration,
		args.center_deadzone_x,
		args.center_deadzone_y,
		args.target_lost_timeout,
	)


if __name__ == '__main__':
	main()
