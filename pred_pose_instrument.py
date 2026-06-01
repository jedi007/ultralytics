from pathlib import Path
import math
from typing import Any

import cv2
import numpy as np

from ultralytics import YOLO


DEFAULT_MODEL_PATH = "pose_instrument_20260528.pt"
DEFAULT_IMAGE_PATH = "test_images/camera_capture_2026-04-24-03-43-31_9305_obj001.jpg"
DEFAULT_OUTPUT_DIR = "runs/pose_demo"
DEFAULT_CONF = 0.01
DEFAULT_IMGSZ = 512
DEFAULT_MIN_VALUE = 0.0
DEFAULT_MAX_VALUE = 100.0
DEFAULT_RECTIFY_METHOD = "auto"
DEFAULT_EXTRA_DETECTED_REF = None
DEFAULT_EXTRA_FRONT_REF = None
DEFAULT_SAVE = False
DEFAULT_NO_SHOW = True
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
	(DEFAULT_FRONT_VIEW_RADIUS, 0.0),
]
print(f"Default front view anchor points: {DEFAULT_FRONT_VIEW_ANCHOR_POINTS}")


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
	"""Calculate a gauge reading from 4 pose keypoints.

	Keypoint order:
	1. center of dial
	2. tip of pointer
	3. minimum-scale point
	4. maximum-scale point
	"""
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
	extra_detected_reference_point: tuple[float, float] | None = None,
	extra_front_view_reference_point: tuple[float, float] | None = None,
) -> dict[str, object]:
	"""Map detected gauge keypoints into a front-view coordinate system.

	Detected keypoints are ordered as:
	1. center of dial
	2. tip of pointer
	3. minimum-scale point
	4. maximum-scale point
	5. top reference point on the dial rim

	Front-view anchor points are ordered as:
	1. center of dial
	2. minimum-scale point
	3. maximum-scale point
	4. top reference point on the dial rim

	Perspective rectification uses detected keypoints 1, 3, 4, 5 mapped to front-view
	anchor points 1, 2, 3, 4. If those points are unavailable, a 3-point affine
	transform is used as a fallback unless method="perspective" is requested.
	"""
	if method not in {"auto", "affine", "perspective"}:
		raise ValueError("method must be one of: auto, affine, perspective")

	requires_perspective_anchors = (
		len(detected_keypoints) >= 5 and len(front_view_anchor_points) >= 4
	)
	detected_required_count = 5 if method == "perspective" or requires_perspective_anchors else 4
	front_view_required_count = 4 if method == "perspective" or requires_perspective_anchors else 3

	detected_points = _points_to_float32(detected_keypoints, detected_required_count, "detected_keypoints")
	front_view_points = _points_to_float32(front_view_anchor_points, front_view_required_count, "front_view_anchor_points")

	source_anchor_points = [detected_points[0], detected_points[2], detected_points[3]]
	destination_anchor_points = front_view_points[:3]

	transform_method = method
	if method == "auto":
		transform_method = "perspective" if requires_perspective_anchors else "affine"

	if transform_method == "perspective":
		if len(detected_points) < 5 or len(front_view_points) < 4:
			raise ValueError(
				"Perspective rectification requires detected keypoints 1,3,4,5 and front-view anchor points 1,2,3,4"
			)
		source_quad = np.array(source_anchor_points + [detected_points[4]], dtype=np.float32)
		destination_quad = np.array(destination_anchor_points + [front_view_points[3]], dtype=np.float32)
		transform_matrix = cv2.getPerspectiveTransform(source_quad, destination_quad)
		transformed = cv2.perspectiveTransform(np.array([detected_points], dtype=np.float32), transform_matrix)[0]
	else:
		transform_matrix = cv2.getAffineTransform(
			np.array(source_anchor_points, dtype=np.float32),
			np.array(destination_anchor_points, dtype=np.float32),
		)
		transformed = cv2.transform(np.array([detected_points], dtype=np.float32), transform_matrix)[0]

	rectified_keypoints = [tuple(float(value) for value in point) for point in transformed]
	return {
		"keypoints": rectified_keypoints,
		"transform_method": transform_method,
		"transform_matrix": transform_matrix,
	}


def calculate_gauge_reading_with_rectification(
	detected_keypoints: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...],
	front_view_anchor_points: list[list[float]] | list[tuple[float, float]] | tuple[tuple[float, float], ...],
	min_value: float = 0.0,
	max_value: float = 100.0,
	method: str = "auto",
	extra_detected_reference_point: tuple[float, float] | None = None,
	extra_front_view_reference_point: tuple[float, float] | None = None,
) -> dict[str, object]:
	"""Rectify detected keypoints to the front view and then calculate the gauge reading."""
	rectification = rectify_gauge_keypoints(
		detected_keypoints=detected_keypoints,
		front_view_anchor_points=front_view_anchor_points,
		method=method,
		extra_detected_reference_point=extra_detected_reference_point,
		extra_front_view_reference_point=extra_front_view_reference_point,
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


class InstrumentGaugeReader:
	def __init__(
		self,
		model_path: str | Path = DEFAULT_MODEL_PATH,
		conf: float = DEFAULT_CONF,
		imgsz: int = DEFAULT_IMGSZ,
		min_value: float = DEFAULT_MIN_VALUE,
		max_value: float = DEFAULT_MAX_VALUE,
		rectify_method: str = DEFAULT_RECTIFY_METHOD,
		front_view_anchor_points: list[tuple[float, float]] | None = None,
		extra_detected_ref: tuple[float, float] | None = None,
		extra_front_ref: tuple[float, float] | None = None,
	) -> None:
		self.model_path = validate_path(str(model_path), "Model file")
		self.conf = conf
		self.imgsz = imgsz
		self.min_value = min_value
		self.max_value = max_value
		self.rectify_method = rectify_method
		self.front_view_anchor_points = front_view_anchor_points or DEFAULT_FRONT_VIEW_ANCHOR_POINTS
		self.extra_detected_ref = extra_detected_ref
		self.extra_front_ref = extra_front_ref
		self.model = YOLO(str(self.model_path))

	def predict(self, image: np.ndarray):
		if image is None or not isinstance(image, np.ndarray):
			raise ValueError("image must be a loaded numpy.ndarray")
		return self.model.predict(source=image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]

	def extract_reading_details(self, result: Any) -> dict[str, Any] | None:
		if result.keypoints is None or result.keypoints.xy is None:
			return None

		for index, pose_keypoints in enumerate(result.keypoints.xy.cpu().tolist(), start=1):
			if len(pose_keypoints) < 4:
				continue
			try:
				gauge_result = calculate_gauge_reading_with_rectification(
					pose_keypoints,
					front_view_anchor_points=self.front_view_anchor_points,
					min_value=self.min_value,
					max_value=self.max_value,
					method=self.rectify_method,
					extra_detected_reference_point=self.extra_detected_ref,
					extra_front_view_reference_point=self.extra_front_ref,
				)
			except ValueError:
				continue

			return {
				"gauge_index": index,
				"result": result,
				"keypoints": pose_keypoints,
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


def validate_path(path_value: str, description: str) -> Path:
	path = Path(path_value)
	if not path.exists():
		raise FileNotFoundError(f"{description} does not exist: {path}")
	return path


def main():
	image_path = validate_path(DEFAULT_IMAGE_PATH, "Image file")
	image = cv2.imread(str(image_path))
	if image is None:
		raise ValueError(f"Failed to read image data from: {image_path}")

	reader = InstrumentGaugeReader(
		model_path=DEFAULT_MODEL_PATH,
		conf=DEFAULT_CONF,
		imgsz=DEFAULT_IMGSZ,
		min_value=DEFAULT_MIN_VALUE,
		max_value=DEFAULT_MAX_VALUE,
		rectify_method=DEFAULT_RECTIFY_METHOD,
		extra_detected_ref=DEFAULT_EXTRA_DETECTED_REF,
		extra_front_ref=DEFAULT_EXTRA_FRONT_REF,
	)
	result = reader.predict(image)
	annotated_image = result.plot()

	boxes = 0 if result.boxes is None else len(result.boxes)
	keypoints = 0 if result.keypoints is None else len(result.keypoints)
	print(f"Inference complete: detected {boxes} object(s), pose set(s): {keypoints}")
	if boxes == 0:
		print(f"No detections passed conf={DEFAULT_CONF:.3f}. Try a lower threshold, e.g. set DEFAULT_CONF = 0.01")
	elif result.boxes is not None and result.boxes.conf is not None:
		confidences = [f"{score:.4f}" for score in result.boxes.conf.cpu().tolist()]
		print(f"Detection confidences: {', '.join(confidences)}")

	reading_details = reader.extract_reading_details(result)
	if reading_details is None:
		print("No valid gauge reading found.")
	else:
		center_x, center_y = (int(round(value)) for value in reading_details["keypoints"][0][:2])
		cv2.putText(
			annotated_image,
			f"Gauge {reading_details['gauge_index']}: {reading_details['reading']:.2f}",
			(center_x + 10, center_y - 10),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 255, 255),
			2,
			cv2.LINE_AA,
		)

		print(
			f"Gauge {reading_details['gauge_index']}: reading={reading_details['reading']:.2f}, "
			f"ratio={reading_details['ratio']:.4f}, "
			f"pointer_angle={reading_details['pointer_angle_deg']:.2f} deg, "
			f"full_scale_angle={reading_details['full_scale_angle_deg']:.2f} deg, "
			f"direction={reading_details['sweep_direction']}"
		)
		if "transform_method" in reading_details:
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
		cv2.imshow("Pose Inference Demo", annotated_image)
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
			"OpenCV display is unavailable in the current environment. Use --no-show or view the saved image."
		) from exc
	finally:
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()