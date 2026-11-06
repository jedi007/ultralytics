#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Switch / indicator-light state detector demo.

The detector runs ``weights/det_button_261105.pt`` on a switch cabinet image and
returns the state of every detected switch / indicator in reading order
(left-to-right and top-to-bottom).  ``values`` holds one integer per detection
in that same order:

* ``1``  - the class name ends with ``-on``
* ``0``  - the class name ends with ``-off``
* ``-1`` - the class name has no on/off suffix (for example ``switch2``)

For a detector result, ``objs`` is sorted in the same reading order as
``values``, so ``objs[i]`` corresponds to ``values[i]``.

Examples:
    python demos/switch_detecter.py
    python demos/switch_detecter.py --source path/to/cabinet.jpg
    python demos/switch_detecter.py --source path/to/cabinet.jpg --show
    python demos/switch_detecter.py --source path/to/video.mp4 --conf 0.4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

DEFAULT_WEIGHTS = ROOT / "weights" / "det_button_261105.pt"
DEFAULT_IMGSZ = (384, 640)
DEFAULT_CONF = 0.5
DEFAULT_ROW_OVERLAP = 0.3
STATE_ON = 1
STATE_OFF = 0
STATE_UNKNOWN = -1
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv"}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_int_bbox(values: Iterable[Any]) -> list[int]:
    return [round(float(value)) for value in values]


def _format_confidence_3dp(confidence: Any) -> float:
    try:
        return round(float(confidence), 3)
    except (TypeError, ValueError):
        return 0.0


def _normalize_names_map(names_map: Any) -> dict[int, str]:
    if isinstance(names_map, dict):
        normalized: dict[int, str] = {}
        for key, value in names_map.items():
            cls_id = _as_int(key, -1)
            if cls_id >= 0:
                normalized[cls_id] = str(value)
        return normalized
    if isinstance(names_map, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names_map)}
    return {}


def _state_from_label(label: str) -> tuple[int, str]:
    """Parse the switch state encoded in a YOLO class name."""
    label_key = label.strip().lower()
    if label_key.endswith("-on"):
        return STATE_ON, "on"
    if label_key.endswith("-off"):
        return STATE_OFF, "off"
    return STATE_UNKNOWN, "unknown"


def _vertical_overlap_ratio(box: Sequence[float], row_bounds: Sequence[float]) -> float:
    """Return the vertical overlap ratio between a box and a row's bounding range."""
    overlap = min(box[3], row_bounds[3]) - max(box[1], row_bounds[1])
    if overlap <= 0:
        return 0.0
    box_height = max(float(box[3]) - float(box[1]), 1.0)
    row_height = max(float(row_bounds[3]) - float(row_bounds[1]), 1.0)
    return overlap / min(box_height, row_height)


def _sort_reading_order(
    detections: list[dict[str, Any]], row_overlap_ratio: float = DEFAULT_ROW_OVERLAP
) -> list[dict[str, Any]]:
    """Sort detections left-to-right and top-to-bottom.

    Boxes whose vertical ranges overlap are treated as one visual row.  This
    keeps a tall switch and the short indicator lights beside it in the same
    row, even when their center ``y`` values differ slightly.
    """
    if len(detections) <= 1:
        return list(detections)

    rows: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: (item["xyxy"][1], item["xyxy"][0], item["xyxy"][2])):
        box = detection["xyxy"]
        best_row: dict[str, Any] | None = None
        best_ratio = 0.0
        for row in rows:
            ratio = _vertical_overlap_ratio(box, row["bounds"])
            if ratio >= row_overlap_ratio and ratio > best_ratio:
                best_row = row
                best_ratio = ratio

        if best_row is None:
            rows.append({"bounds": list(box), "detections": [detection]})
            continue

        best_row["detections"].append(detection)
        bounds = best_row["bounds"]
        best_row["bounds"] = [
            min(bounds[0], box[0]),
            min(bounds[1], box[1]),
            max(bounds[2], box[2]),
            max(bounds[3], box[3]),
        ]

    rows.sort(
        key=lambda row: (
            (row["bounds"][1] + row["bounds"][3]) / 2.0,
            (row["bounds"][0] + row["bounds"][2]) / 2.0,
        )
    )
    return [
        detection
        for row in rows
        for detection in sorted(
            row["detections"],
            key=lambda item: (
                (item["xyxy"][0] + item["xyxy"][2]) / 2.0,
                (item["xyxy"][1] + item["xyxy"][3]) / 2.0,
            ),
        )
    ]


class SwitchDetecter:
    """YOLO-based switch / indicator-light detector.

    The public result mirrors the inspection pipeline detector contract:
    ``cls_id``, ``xywh``, ``xyxy``, ``objs``, ``values``, ``inspection_type``,
    ``error_msg`` and ``instrument_readdata``.  ``values`` and ``objs`` are
    sorted in reading order.
    """

    def __init__(
        self,
        det_switch_config: dict[str, Any] | None = None,
        *,
        env_var: str = "RD_AI_DET_SWITCH_WEIGHTS",
        role: str = "Switch detection YOLO weight",
    ) -> None:
        config = det_switch_config or {}
        raw_labels = config.get("labels", [])
        self.labels = [str(label) for label in raw_labels] if isinstance(raw_labels, (list, tuple)) else []
        self.confidence_threshold = _as_float(config.get("confidence_threshold"), DEFAULT_CONF)
        raw_imgsz = config.get("imgsz", DEFAULT_IMGSZ)
        if raw_imgsz is None:
            self.imgsz = list(DEFAULT_IMGSZ)
        elif isinstance(raw_imgsz, (int, float)):
            self.imgsz = _as_int(raw_imgsz, DEFAULT_IMGSZ[0])
        else:
            self.imgsz = list(raw_imgsz)
        self.min_obj_size = max(0, _as_int(config.get("min_obj_size"), 0))
        self.row_overlap_ratio = max(0.0, _as_float(config.get("row_overlap_ratio"), DEFAULT_ROW_OVERLAP))
        self.model = None
        self._load_error = ""

        weight_file_name = str(config.get("weight_file_name") or os.environ.get(env_var, "") or DEFAULT_WEIGHTS).strip()
        try:
            weight_path = self._resolve_weights(weight_file_name)
            self.model = YOLO(str(weight_path))
        except Exception as exc:  # noqa: BLE001 - surface model/path failures as a load error
            self._load_error = f"{role} could not be loaded: {exc}"
            self.model = None

    @staticmethod
    def _resolve_weights(weight_file_name: str) -> Path:
        path = Path(weight_file_name).expanduser()
        if path.is_absolute():
            if path.is_file():
                return path
            raise FileNotFoundError(f"weight file not found: {path}")

        candidates = [Path.cwd() / path, ROOT / path, FILE.parent / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"weight file not found: {weight_file_name} (tried: {tried})")

    @staticmethod
    def _empty_result(item: dict[str, Any], item_type: str | None, error_msg: str) -> dict[str, Any]:
        return {
            "cls_id": -1,
            "xywh": [0, 0, 0, 0],
            "xyxy": [0, 0, 0, 0],
            "objs": [{"confidence": 0.0, "bbox": [0, 0, 0, 0], "label": item_type}],
            "inspection_type": item.get("inspection_type", None),
            "error_msg": error_msg,
            "values": [],
            "instrument_readdata": None,
            "is_capture_only": False,
        }

    def _label_name(self, names_map: dict[int, str], cls_id: int, item_type: str | None) -> str:
        label = names_map.get(cls_id)
        if label:
            return label
        if 0 <= cls_id < len(self.labels):
            return self.labels[cls_id]
        return item_type or "switch"

    def detect(self, image: Any, item: dict[str, Any] | None = None, item_type: str | None = None) -> dict[str, Any]:
        """Detect switches / indicator lights and return states in reading order."""
        item = item if isinstance(item, dict) else {}
        item_result = self._empty_result(item, item_type, "")
        if self.model is None:
            item_result.update(
                {
                    "is_capture_only": True,
                    "error_msg": f"Switch detection model not loaded: {self._load_error}",
                }
            )
            return item_result

        try:
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - surface inference failures in the result payload
            item_result.update(
                {
                    "is_capture_only": True,
                    "error_msg": f"Switch detection failed: {exc}",
                }
            )
            return item_result

        if not results:
            return self._empty_result(item, item_type, "No switch detected.")

        det_result = results[0]
        boxes = getattr(det_result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return self._empty_result(item, item_type, "No switch detected.")

        names_map = _normalize_names_map(getattr(self.model, "names", None) or getattr(det_result, "names", None))

        detections: list[dict[str, Any]] = []
        skipped_by_size = 0
        for index in range(len(boxes)):
            box = boxes[index]
            try:
                cls_id = int(box.cls.item())
                confidence = _format_confidence_3dp(box.conf.item())
                # Ultralytics xywh is (center_x, center_y, width, height). The
                # result contract uses (x, y, width, height), so derive it from xyxy.
                xyxy = _as_int_bbox(box.xyxy.tolist()[0])
                xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
            except (AttributeError, IndexError, TypeError, ValueError):
                continue

            if self.min_obj_size > 0 and (xywh[2] < self.min_obj_size or xywh[3] < self.min_obj_size):
                skipped_by_size += 1
                continue

            label_name = self._label_name(names_map, cls_id, item_type)
            state, state_name = _state_from_label(label_name)
            detections.append(
                {
                    "cls_id": cls_id,
                    "confidence": confidence,
                    "xywh": xywh,
                    "xyxy": xyxy,
                    "label": label_name,
                    "state": state,
                    "state_name": state_name,
                }
            )

        if not detections:
            if skipped_by_size > 0:
                error_msg = f"No switch detected: all candidates below min_obj_size={self.min_obj_size}px."
            else:
                error_msg = "No switch detected."
            return self._empty_result(item, item_type, error_msg)

        detections = _sort_reading_order(detections, self.row_overlap_ratio)
        values = [detection["state"] for detection in detections]
        objs = [
            {
                "confidence": detection["confidence"],
                "bbox": detection["xywh"],
                "label": detection["label"],
                "state": detection["state_name"],
            }
            for detection in detections
        ]

        # The result describes the whole switch box, so its bbox is the union of
        # all detected components while objs keeps each component separately.
        xyxy = [
            min(detection["xyxy"][0] for detection in detections),
            min(detection["xyxy"][1] for detection in detections),
            max(detection["xyxy"][2] for detection in detections),
            max(detection["xyxy"][3] for detection in detections),
        ]
        xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]

        return {
            "cls_id": -1,
            "xywh": xywh,
            "xyxy": xyxy,
            "instrument_readdata": None,
            "values": values,
            "objs": objs,
            "inspection_type": item.get("inspection_type", None),
            "error_msg": "",
        }

    def det_switch(
        self, image: Any, item: dict[str, Any] | None = None, item_type: str | None = None
    ) -> dict[str, Any]:
        """Alias matching the naming convention used by ``SmokingDetecter``."""
        return self.detect(image, item, item_type)


def _print_result(source: str, result: dict[str, Any]) -> None:
    print(f"{source}: values={result['values']}")
    if result.get("error_msg"):
        print(f"  error: {result['error_msg']}")
    for index, obj in enumerate(result.get("objs", []), start=1):
        bbox = obj.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        print(
            f"  {index:>2}. {obj.get('label', ''):<16} "
            f"state={obj.get('state', 'unknown'):<7} "
            f"bbox={bbox} conf={obj.get('confidence', 0.0):.3f}"
        )


def _state_color(state: int) -> tuple[int, int, int]:
    if state == STATE_ON:
        return 0, 200, 0
    if state == STATE_OFF:
        return 0, 0, 255
    return 0, 215, 255


def _draw_result(image: Any, result: dict[str, Any]) -> Any:
    vis_image = image.copy()
    values = result.get("values", [])
    objs = result.get("objs", [])
    height, width = vis_image.shape[:2]
    scale = max(0.4, min(1.0, min(height, width) / 1500.0))
    thickness = max(1, round(scale * 2))

    for index, obj in enumerate(objs):
        bbox = obj.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        x, y, box_width, box_height = [int(value) for value in bbox]
        if box_width <= 0 or box_height <= 0:
            continue

        state = values[index] if index < len(values) else STATE_UNKNOWN
        color = _state_color(state)
        cv2.rectangle(vis_image, (x, y), (x + box_width, y + box_height), color, thickness)

        # Draw only the reading-order index on the image; the full label and
        # state are printed to the console to keep dense switch boxes readable.
        tag = str(index + 1)
        (tag_width, tag_height), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        tag_y = max(y, tag_height + baseline + 2)
        cv2.rectangle(
            vis_image,
            (x, tag_y - tag_height - baseline - 2),
            (x + tag_width + 6, tag_y + baseline - 2),
            color,
            -1,
        )
        cv2.putText(
            vis_image,
            tag,
            (x + 3, tag_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    # Show the values list as a banner, wrapped to fit the image width.
    if values:
        chunk_size = 12
        chunks = [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]
        line_height = round(scale * 32)
        banner_height = line_height * len(chunks) + 10
        cv2.rectangle(vis_image, (0, 0), (width, banner_height), (0, 0, 0), -1)
        for row, chunk in enumerate(chunks):
            prefix = "values=" if row == 0 else "       "
            cv2.putText(
                vis_image,
                f"{prefix}{chunk}",
                (10, 8 + line_height * (row + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
    return vis_image


def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES)


def _process_image(path: Path, detector: SwitchDetecter, save_dir: Path, show: bool) -> None:
    image = cv2.imread(str(path))
    if image is None:
        print(f"failed to read image: {path}", file=sys.stderr)
        return

    result = detector.detect(image, {"inspection_type": "switch_state"}, "switch")
    _print_result(str(path), result)

    vis_image = _draw_result(image, result)
    output_path = save_dir / f"{path.stem}_switch{path.suffix.lower()}"
    if not cv2.imwrite(str(output_path), vis_image):
        print(f"failed to save result: {output_path}", file=sys.stderr)
    else:
        print(f"  saved: {output_path}")

    if show:
        cv2.imshow("SwitchDetecter", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _process_stream(source: str, detector: SwitchDetecter, save_dir: Path, show: bool) -> None:
    capture_source: Any = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        print(f"failed to open source: {source}", file=sys.stderr)
        if str(source).isdigit():
            print(
                "No camera was found at that index. Use --source /path/to/image_or_video instead.",
                file=sys.stderr,
            )
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    output_name = "webcam_switch.mp4" if source.isdigit() else f"{Path(source).stem}_switch.mp4"
    output_path = save_dir / output_name
    writer: cv2.VideoWriter | None = None
    last_values: list[int] | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            result = detector.detect(frame, {"inspection_type": "switch_state"}, "switch")
            if result["values"] != last_values:
                last_values = result["values"]
                _print_result(source, result)

            vis_image = _draw_result(frame, result)
            if writer is None:
                height, width = vis_image.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    print(f"failed to create video: {output_path}", file=sys.stderr)
                    writer = None
            if writer is not None:
                writer.write(vis_image)

            if show:
                cv2.imshow("SwitchDetecter", vis_image)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"  saved: {output_path}")
        if show:
            cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Switch / indicator-light state detector demo.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="path to the switch detection weights")
    parser.add_argument(
        "--source",
        default=None,
        help="image, directory, video, stream URL, or camera index; defaults to DEFAULT_SOURCE in main()",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=list(DEFAULT_IMGSZ),
        metavar=("H", "W"),
        help="inference size as height width (default: 384 640)",
    )
    parser.add_argument("--min-obj-size", type=int, default=0, help="drop boxes smaller than this size in pixels")
    parser.add_argument(
        "--row-overlap",
        type=float,
        default=DEFAULT_ROW_OVERLAP,
        help="minimum vertical overlap ratio used to group detections into rows",
    )
    parser.add_argument("--save-dir", default=str(ROOT / "runs" / "switch_demo"), help="output directory")
    parser.add_argument("--show", action="store_true", help="display results in a window")
    return parser.parse_args()


def main() -> int:
    # 未传入 --source 时使用的默认测试图片，按需修改此路径。
    DEFAULT_SOURCE = "test_images/switch_test1.jpg"

    args = _parse_args()
    detector = SwitchDetecter(
        {
            "weight_file_name": args.weights,
            "confidence_threshold": args.conf,
            "imgsz": args.imgsz,
            "min_obj_size": args.min_obj_size,
            "row_overlap_ratio": args.row_overlap,
        }
    )
    if detector.model is None:
        print(detector._load_error, file=sys.stderr)
        return 1

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    source = str(args.source if args.source is not None else DEFAULT_SOURCE)

    if source.isdigit():
        _process_stream(source, detector, save_dir, args.show)
        return 0

    source_path = Path(source).expanduser()
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
        _process_image(source_path, detector, save_dir, args.show)
    elif source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES:
        _process_stream(source, detector, save_dir, args.show)
    elif source_path.is_dir():
        images = _collect_images(source_path)
        if not images:
            print(f"no images found in: {source_path}", file=sys.stderr)
            return 1
        for image_path in images:
            _process_image(image_path, detector, save_dir, args.show)
    else:
        # Let OpenCV handle RTSP/RTMP/HTTP streams and any other video source.
        _process_stream(source, detector, save_dir, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
