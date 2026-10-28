from pathlib import Path
import math
from typing import Any

import cv2
import numpy as np

from ultralytics import YOLO


DEFAULT_MODEL_PATH = "weights/det_instrument_pointer_20261022.pt"
DEFAULT_IMAGE_PATH = "test_images/camera_capture_2026-04-24-03-43-31_9305_obj002.jpg"
DEFAULT_OUTPUT_DIR = "runs/det_pointer_demo"
DEFAULT_CONF = 0.25
DEFAULT_GAUGE_CONF = 0.25
DEFAULT_IMGSZ = 512
DEFAULT_MIN_VALUE = 0.0
DEFAULT_MAX_VALUE = 100.0
DEFAULT_RECTIFY_METHOD = "auto"
DEFAULT_SAVE = False
DEFAULT_NO_SHOW = False
DEFAULT_FRONT_VIEW_RADIUS = 256.0
DEFAULT_FRONT_VIEW_ANCHOR_POINTS = [
	(DEFAULT_FRONT_VIEW_RADIUS, DEFAULT_FRONT_VIEW_RADIUS),
	(
		DEFAULT_FRONT_VIEW_RADIUS * (1.0 - math.sqrt(0.5)),
		DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sqrt(0.5)),
	),
	(
		DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sqrt(0.5)),
		DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sqrt(0.5)),
	),
]
DEFAULT_CLASS_ALIASES = {
	"gauge": ("gauge", "instrument", "meter"),
	"center": ("center", "centre"),
	"tip": ("tip", "pointer", "needle"),
	"max": ("max", "maximum"),
	"min": ("min", "minimum"),
}
ROLE_ORDER = ("center", "tip", "min", "max")
ROLE_COLORS = {
	"gauge": (255, 0, 255),
	"center": (0, 255, 0),
	"tip": (0, 0, 255),
	"min": (255, 0, 0),
	"max": (0, 255, 255),
}


def _normalize_angle(angle_deg: float) -> float:
	return angle_deg % 360.0


def _point_angle_deg(center: tuple[float, float], point: tuple[float, float]) -> float:
	return _normalize_angle(math.degrees(math.atan2(point[1] - center[1], point[0] - center[0])))


def _clockwise_delta_deg(start_deg: float, end_deg: float) -> float:
	return _normalize_angle(end_deg - start_deg)


def _counterclockwise_delta_deg(start_deg: float, end_deg: float) -> float:
	return _normalize_angle(start_deg - end_deg)


def calculate_gauge_reading(
	center: tuple[float, float],
	pointer_tip: tuple[float, float],
	min_point: tuple[float, float],
	max_point: tuple[float, float],
	min_value: float = 0.0,
	max_value: float = 100.0,
) -> dict[str, float | str]:
	"""Calculate a gauge reading from the center, pointer tip, min point and max point."""
	if max_value <= min_value:
		raise ValueError("max_value must be greater than min_value")

	min_angle = _point_angle_deg(center, min_point)
	max_angle = _point_angle_deg(center, max_point)
	pointer_angle = _point_angle_deg(center, pointer_tip)

	cw_range = _clockwise_delta_deg(min_angle, max_angle)
	cw_pointer = _clockwise_delta_deg(min_angle, pointer_angle)
	ccw_range = _counterclockwise_delta_deg(min_angle, max_angle)
	ccw_pointer = _counterclockwise_delta_deg(min_angle, pointer_angle)

	if cw_range == 0.0 or ccw_range == 0.0:
		raise ValueError("Minimum point and maximum point must not overlap")

	if cw_pointer <= cw_range:
		sweep_direction = "clockwise"
		full_scale_angle = cw_range
		pointer_angle_from_min = cw_pointer
	else:
		sweep_direction = "counterclockwise"
		full_scale_angle = ccw_range
		pointer_angle_from_min = min(ccw_pointer, ccw_range)

	ratio = 0.0 if full_scale_angle == 0.0 else pointer_angle_from_min / full_scale_angle
	ratio = max(0.0, min(1.0, ratio))
	reading = min_value + ratio * (max_value - min_value)

	return {
		"reading": reading,
		"percentage": ratio * 100.0,
		"ratio": ratio,
		"pointer_angle_deg": pointer_angle_from_min,
		"full_scale_angle_deg": full_scale_angle,
		"sweep_direction": sweep_direction,
		"min_angle_deg": min_angle,
		"max_angle_deg": max_angle,
		"pointer_absolute_angle_deg": pointer_angle,
	}


def calculate_gauge_reading_from_keypoints(
	keypoints: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...],
	min_value: float = 0.0,
	max_value: float = 100.0,
) -> dict[str, float | str]:
	if len(keypoints) < 4:
		raise ValueError("At least 4 keypoints are required: center, pointer tip, min point, max point")

	center = tuple(float(value) for value in keypoints[0][:2])
	pointer_tip = tuple(float(value) for value in keypoints[1][:2])
	min_point = tuple(float(value) for value in keypoints[2][:2])
	max_point = tuple(float(value) for value in keypoints[3][:2])
	return calculate_gauge_reading(center, pointer_tip, min_point, max_point, min_value=min_value, max_value=max_value)


def _points_to_float32(
	points: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...],
	required_count: int,
	label: str,
) -> list[tuple[float, float]]:
	if len(points) < required_count:
		raise ValueError(f"{label} must contain at least {required_count} points")
	return [tuple(float(value) for value in point[:2]) for point in points[:required_count]]


def rectify_gauge_keypoints(
	detected_keypoints: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...],
	front_view_anchor_points: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...],
	method: str = "auto",
) -> dict[str, object]:
	"""Map detected center, tip, min and max points into a front-view coordinate system."""
	if method not in {"auto", "affine"}:
		raise ValueError("method must be one of: auto, affine")

	detected_points = _points_to_float32(detected_keypoints, 4, "detected_keypoints")
	front_view_points = _points_to_float32(front_view_anchor_points, 3, "front_view_anchor_points")

	source_anchor_points = np.array(
		[detected_points[0], detected_points[2], detected_points[3]],
		dtype=np.float32,
	)
	destination_anchor_points = np.array(front_view_points, dtype=np.float32)
	transform_matrix = cv2.getAffineTransform(source_anchor_points, destination_anchor_points)
	transformed = cv2.transform(np.array([detected_points], dtype=np.float32), transform_matrix)[0]

	rectified_keypoints = [tuple(float(value) for value in point) for point in transformed]
	return {
		"keypoints": rectified_keypoints,
		"transform_method": "affine",
		"transform_matrix": transform_matrix,
	}


def calculate_gauge_reading_with_rectification(
	detected_keypoints: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...],
	front_view_anchor_points: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...],
	min_value: float = 0.0,
	max_value: float = 100.0,
	method: str = "auto",
) -> dict[str, object]:
	"""Rectify detected keypoints to the front view and then calculate the gauge reading."""
	rectification = rectify_gauge_keypoints(
		detected_keypoints=detected_keypoints,
		front_view_anchor_points=front_view_anchor_points,
		method=method,
	)
	reading = calculate_gauge_reading_from_keypoints(
		rectification["keypoints"],
		min_value=min_value,
		max_value=max_value,
	)
	return {
		**reading,
		"rectified_keypoints": rectification["keypoints"],
		"transform_method": rectification["transform_method"],
		"transform_matrix": rectification["transform_matrix"],
	}


def _find_class_ids(names: Any, aliases: tuple[str, ...]) -> set[int]:
	if isinstance(names, dict):
		items = names.items()
	else:
		items = enumerate(names)
	normalized_aliases = {alias.lower() for alias in aliases}
	return {
		int(class_id)
		for class_id, name in items
		if str(name).lower() in normalized_aliases
	}


def _box_center(xyxy: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
	x1, y1, x2, y2 = (float(value) for value in xyxy[:4])
	return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_inside_box(
	point: tuple[float, float],
	xyxy: list[float] | tuple[float, float, float, float],
	margin_ratio: float = 0.1,
) -> bool:
	x1, y1, x2, y2 = (float(value) for value in xyxy[:4])
	margin_x = max(0.0, x2 - x1) * margin_ratio
	margin_y = max(0.0, y2 - y1) * margin_ratio
	return x1 - margin_x <= point[0] <= x2 + margin_x and y1 - margin_y <= point[1] <= y2 + margin_y


class DetectionGaugeReader:
	def __init__(
		self,
		model_path: str | Path = DEFAULT_MODEL_PATH,
		conf: float = DEFAULT_CONF,
		gauge_conf: float = DEFAULT_GAUGE_CONF,
		imgsz: int = DEFAULT_IMGSZ,
		min_value: float = DEFAULT_MIN_VALUE,
		max_value: float = DEFAULT_MAX_VALUE,
		rectify_method: str = DEFAULT_RECTIFY_METHOD,
		front_view_anchor_points: list[tuple[float, float]] | None = None,
		class_aliases: dict[str, tuple[str, ...]] | None = None,
	) -> None:
		self.model_path = validate_path(str(model_path), "Model file")
		self.conf = conf
		self.gauge_conf = gauge_conf
		self.imgsz = imgsz
		self.min_value = min_value
		self.max_value = max_value
		self.rectify_method = rectify_method
		self.front_view_anchor_points = front_view_anchor_points or DEFAULT_FRONT_VIEW_ANCHOR_POINTS
		self.class_aliases = class_aliases or DEFAULT_CLASS_ALIASES
		self.model = YOLO(str(self.model_path))
		self.class_ids = {
			role: _find_class_ids(self.model.names, aliases) for role, aliases in self.class_aliases.items()
		}
		missing_roles = [role for role, ids in self.class_ids.items() if not ids]
		if missing_roles:
			raise ValueError(f"Model is missing required classes: {', '.join(missing_roles)}; model names: {self.model.names}")
		self.last_result: Any | None = None
		self.last_reading_details: dict[str, Any] | None = None

	def predict(self, image: np.ndarray):
		if image is None or not isinstance(image, np.ndarray):
			raise ValueError("image must be a loaded numpy.ndarray")
		return self.model.predict(source=image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]

	def _extract_detections(self, result: Any) -> list[dict[str, Any]]:
		if result.boxes is None or len(result.boxes) == 0:
			return []

		xyxy_values = result.boxes.xyxy.cpu().tolist()
		confidence_values = result.boxes.conf.cpu().tolist()
		class_values = result.boxes.cls.int().cpu().tolist()
		detections = []
		for index, (xyxy, confidence, class_id) in enumerate(
			zip(xyxy_values, confidence_values, class_values)
		):
			box = [float(value) for value in xyxy]
			detections.append(
				{
					"index": index,
					"xyxy": box,
					"center": _box_center(box),
					"confidence": float(confidence),
					"class_id": int(class_id),
					"class_name": str(result.names[int(class_id)]),
				}
			)
		return detections

	def extract_reading_details(self, result: Any) -> dict[str, Any] | None:
		detections = self._extract_detections(result)
		if not detections:
			return None

		gauge_detections = [
			detection
			for detection in detections
			if detection["class_id"] in self.class_ids["gauge"] and detection["confidence"] >= self.gauge_conf
		]
		gauge_detections.sort(key=lambda detection: detection["confidence"], reverse=True)
		if not gauge_detections:
			return None

		for gauge_index, gauge_detection in enumerate(gauge_detections, start=1):
			candidate_detections = [
				detection
				for detection in detections
				if detection["class_id"] != gauge_detection["class_id"]
				and _point_inside_box(detection["center"], gauge_detection["xyxy"], margin_ratio=0.15)
			]
			selected_detections: dict[str, dict[str, Any]] = {}
			for role in ROLE_ORDER:
				role_detections = [
					detection
					for detection in candidate_detections
					if detection["class_id"] in self.class_ids[role]
				]
				if not role_detections:
					continue
				selected_detections[role] = max(role_detections, key=lambda detection: detection["confidence"])

			if len(selected_detections) != len(ROLE_ORDER):
				continue

			keypoints = [selected_detections[role]["center"] for role in ROLE_ORDER]
			try:
				gauge_result = calculate_gauge_reading_with_rectification(
					keypoints,
					front_view_anchor_points=self.front_view_anchor_points,
					min_value=self.min_value,
					max_value=self.max_value,
					method=self.rectify_method,
				)
			except (ValueError, cv2.error):
				continue

			return {
				"gauge_index": gauge_index,
				"confidence": gauge_detection["confidence"],
				"keypoint_confidences": {
					role: selected_detections[role]["confidence"] for role in ROLE_ORDER
				},
				"gauge_box": gauge_detection["xyxy"],
				"keypoint_boxes": {
					role: selected_detections[role]["xyxy"] for role in ROLE_ORDER
				},
				"result": result,
				"keypoints": keypoints,
				**gauge_result,
			}

		return None

	def read_image_details(self, image: np.ndarray) -> dict[str, Any] | None:
		result = self.predict(image)
		return self.extract_reading_details(result)

	def read_image(self, image: np.ndarray) -> float | None:
		reading_details = self.read_image_details(image)
		if reading_details is None:
			return None
		return float(reading_details["reading"])

	def forward(self, image: np.ndarray, min_value: float, max_value: float) -> tuple[float | None, float | None]:
		result = self.predict(image)
		self.last_result = result
		self.min_value = min_value
		self.max_value = max_value
		reading_details = self.extract_reading_details(result)
		self.last_reading_details = reading_details
		if reading_details is None:
			return None, None
		return float(reading_details["reading"]), float(reading_details["confidence"])


def validate_path(path_value: str, description: str) -> Path:
	path = Path(path_value)
	if not path.exists():
		raise FileNotFoundError(f"{description} does not exist: {path}")
	return path


def _draw_reading_details(image: np.ndarray, details: dict[str, Any]) -> np.ndarray:
	annotated = image.copy()
	for role, box in details["keypoint_boxes"].items():
		x1, y1, x2, y2 = (int(round(value)) for value in box)
		color = ROLE_COLORS[role]
		cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
		cx, cy = (int(round(value)) for value in _box_center(box))
		cv2.circle(annotated, (cx, cy), 3, color, -1)

	center = tuple(int(round(value)) for value in details["keypoints"][0])
	tip = tuple(int(round(value)) for value in details["keypoints"][1])
	min_point = tuple(int(round(value)) for value in details["keypoints"][2])
	max_point = tuple(int(round(value)) for value in details["keypoints"][3])
	cv2.line(annotated, center, min_point, ROLE_COLORS["min"], 2)
	cv2.line(annotated, center, max_point, ROLE_COLORS["max"], 2)
	cv2.line(annotated, center, tip, ROLE_COLORS["tip"], 2)

	text = f"Gauge {details['gauge_index']}: {details['percentage']:.2f}%"
	text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
	text_x = 10
	text_y = annotated.shape[0] - 15
	cv2.rectangle(
		annotated,
		(text_x - 5, text_y - text_size[1] - baseline - 4),
		(text_x + text_size[0] + 5, text_y + baseline),
		(0, 0, 0),
		-1,
	)
	cv2.putText(
		annotated,
		text,
		(text_x, text_y),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		ROLE_COLORS["gauge"],
		2,
		cv2.LINE_AA,
	)
	return annotated


def main():
	image_path = validate_path(DEFAULT_IMAGE_PATH, "Image file")
	image = cv2.imread(str(image_path))
	if image is None:
		raise ValueError(f"Failed to read image data from: {image_path}")

	reader = DetectionGaugeReader(
		model_path=DEFAULT_MODEL_PATH,
		conf=DEFAULT_CONF,
		gauge_conf=DEFAULT_GAUGE_CONF,
		imgsz=DEFAULT_IMGSZ,
		min_value=DEFAULT_MIN_VALUE,
		max_value=DEFAULT_MAX_VALUE,
		rectify_method=DEFAULT_RECTIFY_METHOD,
	)
	result = reader.predict(image)
	reader.last_result = result
	reading_details = reader.extract_reading_details(result)
	reader.last_reading_details = reading_details

	annotated_image = result.plot(conf=True, labels=True, boxes=True)
	boxes = 0 if result.boxes is None else len(result.boxes)
	print(f"Inference complete: detected {boxes} object(s)")
	if boxes == 0:
		print(f"No detections passed conf={DEFAULT_CONF:.3f}.")
	elif result.boxes is not None and result.boxes.conf is not None:
		confidences = [f"{score:.4f}" for score in result.boxes.conf.cpu().tolist()]
		print(f"Detection confidences: {', '.join(confidences)}")

	if reading_details is None:
		print(
			f"No valid gauge reading found. Ensure class 'gauge' passed gauge_conf={DEFAULT_GAUGE_CONF:.3f} "
			f"and center, tip/pointer, min and max are detected inside the gauge box."
		)
	else:
		annotated_image = _draw_reading_details(annotated_image, reading_details)
		print(
			f"Gauge {reading_details['gauge_index']}: reading={reading_details['reading']:.2f}, "
			f"percentage={reading_details['percentage']:.2f}%, "
			f"confidence={reading_details['confidence']:.4f}, "
			f"ratio={reading_details['ratio']:.4f}, "
			f"pointer_angle={reading_details['pointer_angle_deg']:.2f} deg, "
			f"full_scale_angle={reading_details['full_scale_angle_deg']:.2f} deg, "
			f"direction={reading_details['sweep_direction']}"
		)
		print(f"Gauge {reading_details['gauge_index']}: rectified with {reading_details['transform_method']} transform")

	output_path = None
	if DEFAULT_SAVE or DEFAULT_NO_SHOW:
		output_dir = Path(DEFAULT_OUTPUT_DIR)
		output_dir.mkdir(parents=True, exist_ok=True)
		output_path = output_dir / f"{image_path.stem}_pred.jpg"
		cv2.imwrite(str(output_path), annotated_image)
		print(f"Annotated image saved to: {output_path}")

	if DEFAULT_NO_SHOW:
		return

	try:
		cv2.imshow("Detection Gauge Reading Demo", annotated_image)
		print("Press any key in the image window to close.")
		cv2.waitKey(0)
	except cv2.error as exc:
		if output_path is None:
			output_dir = Path(DEFAULT_OUTPUT_DIR)
			output_dir.mkdir(parents=True, exist_ok=True)
			output_path = output_dir / f"{image_path.stem}_pred.jpg"
			cv2.imwrite(str(output_path), annotated_image)
			print(f"Display unavailable, annotated image saved to: {output_path}")
		raise RuntimeError(
			"OpenCV display is unavailable in the current environment. Use DEFAULT_NO_SHOW = True to view the saved image."
		) from exc
	finally:
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
