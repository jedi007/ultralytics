#!/usr/bin/env python3
"""使用 OpenCV 从 USB 摄像头拉流并显示。"""

import argparse
import subprocess
import sys
from datetime import datetime
import time

import cv2


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="USB 摄像头拉流显示")
	parser.add_argument("--camera", type=int, default=0, help="摄像头索引，默认 0")
	parser.add_argument("--width", type=int, default=1920, help="期望分辨率宽")
	parser.add_argument("--height", type=int, default=1080, help="期望分辨率高")
	parser.add_argument("--fps", type=int, default=30, help="期望帧率")
	parser.add_argument("--window", type=str, default="USB Camera", help="显示窗口名")
	parser.add_argument(
		"--sharpness",
		type=int,
		default=200,
		help="摄像头锐度(0-2048)，默认 200；设为 -1 跳过设置",
	)
	parser.add_argument(
		"--display-scale",
		type=float,
		default=1.0,
		help="显示缩放比例，默认 1.0；大于 1 放大，小于 1 缩小",
	)
	parser.add_argument(
		"--denoise",
		type=str,
		choices=["none", "gaussian", "bilateral"],
		default="none",
		help="显示前降噪方式，默认 none",
	)
	return parser.parse_args()


def try_set_camera_sharpness(camera_id: int, sharpness: int) -> None:
	if sharpness < 0:
		return

	device_path = f"/dev/video{camera_id}"
	result = subprocess.run(
		["v4l2-ctl", "-d", device_path, "-c", f"sharpness={sharpness}"],
		capture_output=True,
		text=True,
	)
	if result.returncode == 0:
		print(f"已设置摄像头锐度 sharpness={sharpness}")
	else:
		message = (result.stderr or result.stdout).strip()
		print(f"提示: 设置锐度失败，继续使用默认参数。{message}")


def open_camera(camera_id: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
	mjpg_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
	last_error = None

	for backend in backends:
		cap = cv2.VideoCapture(camera_id, backend)
		if not cap.isOpened():
			last_error = f"backend={backend} 打开失败"
			continue

		# 先切到压缩格式，再设置分辨率和帧率，避免后续格式切换覆盖帧率设置。
		cap.set(cv2.CAP_PROP_FOURCC, mjpg_fourcc)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
		cap.set(cv2.CAP_PROP_FPS, fps)
		# 某些 UVC 设备在分辨率切换后会重置帧率，再写一遍提高生效率。
		cap.set(cv2.CAP_PROP_FPS, fps)
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

		actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		actual_fps = cap.get(cv2.CAP_PROP_FPS)
		actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
		actual_fourcc_text = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))

		print(
			"摄像头打开成功: "
			f"backend={backend}, request={width}x{height}@{fps}, "
			f"actual={actual_width}x{actual_height}@{actual_fps:.1f}, fourcc={actual_fourcc_text}"
		)

		if actual_fps > 0 and abs(actual_fps - fps) > 0.5:
			print(
				"警告: 设备未接受目标帧率，"
				f"请求 {fps} FPS，实际协商 {actual_fps:.1f} FPS。"
			)

		for _ in range(5):
			cap.grab()

		return cap

	raise RuntimeError(f"无法打开 USB 摄像头，camera={camera_id}, {last_error}")


class CameraCapture:
	"""封装摄像头拉流为一个可复用类。"""

	def __init__(self, camera: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
		self.camera = camera
		self.width = width
		self.height = height
		self.fps = fps
		self.cap: cv2.VideoCapture | None = None

	def open(self) -> None:
		"""打开摄像头（使用现有的 open_camera helper）。"""
		self.cap = open_camera(self.camera, self.width, self.height, self.fps)

	def read(self) -> tuple[bool, object]:
		"""读取一帧，返回 (ok, frame)。"""
		if self.cap is None:
			return False, None
		return self.cap.read()

	def release(self) -> None:
		if self.cap is not None:
			try:
				self.cap.release()
			finally:
				self.cap = None



def main() -> int:
	args = parse_args()
	try_set_camera_sharpness(args.camera, args.sharpness)


	cam = CameraCapture(args.camera, args.width, args.height, args.fps)
	try:
		cam.open()
	except RuntimeError as exc:
		print(exc)
		return 1

	cv2.namedWindow(args.window, cv2.WINDOW_AUTOSIZE)

	print("摄像头已打开：按 q 或 ESC 退出")
	last_time = time.time()
	fps = 0.0
	try:
		while True:
			ok, frame = cam.read()
			if not ok:
				print("读取视频帧失败，退出。")
				break

			if args.denoise == "gaussian":
				frame = cv2.GaussianBlur(frame, (3, 3), 0)
			elif args.denoise == "bilateral":
				frame = cv2.bilateralFilter(frame, 5, 35, 35)

			now = time.time()
			delta = now - last_time
			if delta > 0:
				fps = 1.0 / delta
			last_time = now

			ts_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			fps_text = f"FPS: {fps:.1f}"

			cv2.putText(
				frame,
				ts_text,
				(16, 32),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 255, 0),
				2,
				cv2.LINE_AA,
			)
			cv2.putText(
				frame,
				fps_text,
				(16, 64),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 255, 0),
				2,
				cv2.LINE_AA,
			)

			if args.display_scale != 1.0:
				new_w = max(1, int(frame.shape[1] * args.display_scale))
				new_h = max(1, int(frame.shape[0] * args.display_scale))
				if args.display_scale > 1.0:
					interp = cv2.INTER_CUBIC
				else:
					interp = cv2.INTER_AREA
				frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)

			cv2.imshow(args.window, frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q") or key == 27:
				break
	finally:
		cam.release()
		cv2.destroyAllWindows()

	return 0


if __name__ == "__main__":
	sys.exit(main())
