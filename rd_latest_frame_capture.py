#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading
import time

import cv2


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
