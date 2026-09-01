from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

import cv2

from point_data_parser import DEFAULT_FRONT_VIEW_ANCHOR_POINTS, parse_readdata
from rd_latest_frame_capture import LatestFrameCapture
from ultralytics import YOLO
from ultralytics.utils import TQDM


# =========================
# Config (edit as needed)
# =========================
MODEL_PATH = "weights/det_instrument_20260817.pt"
RTSP_URL = "rtsp://admin:sshw1234@10.42.0.30:554/cam/realmonitor?channel=1&subtype=0"
RTSP_TRANSPORT = "tcp"  # RTSP 传输协议，可选 "tcp" 或 "udp"
CONF = 0.55
IOU = 0.45
IMGSZ = [384, 640]
DEVICE = None
WINDOW_NAME = "YOLO Real-time Detection"
QUIT_KEY = "q"  # 按 q 退出
DISPLAY_SCALE = 1.0  # 显示窗口相对原图的缩放比例
MAX_DISPLAY_FPS = 10.0  # 显示帧率上限，避免播放过快

# PTZ 配置 (用于获取实时 zoom_step)
PTZ_HOST = "10.42.0.30"
PTZ_PORT = 37777
PTZ_USERNAME = "admin"
PTZ_PASSWORD = "sshw1234"
PTZ_CHANNEL = 0

# 单目测距参数
TARGET_SIZE_MM = 92.0    # 目标实际大小 (mm)
TARGET_SIZE_MM_BY_LABEL: dict[str, float] = {
    "instrument": 92.0,
    "instrument_led": 52.0,
}

F_DIV_SENSOR_BY_ZOOM = {
    1: 2.0847,
    2: 2.2100,
    3: 2.3923,
    4: 2.7682,
    5: 2.9885,
    6: 3.3416,
    7: 3.5998,
    8: 3.8884,
    9: 4.0631,
    10: 4.3669,
    11: 4.6023,
    12: 4.9213,
    13: 5.2137,
    14: 5.4301,
    15: 5.6504,
    16: 5.8786,
    17: 6.1056,
    18: 6.4709,
    19: 6.6259,
    20: 7.0798,
    21: 7.2459,
    22: 7.5559,
    23: 7.7939,
    24: 8.1371,
    25: 8.3585,
    26: 8.5799,
    27: 8.8788,
    28: 9.0836,
    29: 9.5099,
    30: 9.7091,
    31: 10.0302,
    32: 10.2350,
    33: 10.4675,
    34: 10.6391,
    35: 11.0432,
    36: 11.2978,
    37: 11.6687,
    38: 11.8237,
    39: 12.1890,
    40: 12.2610,
    41: 12.6706,
    42: 12.8920,
    43: 13.1577,
    44: 13.3514,
    45: 13.6171,
    46: 13.9548,
    47: 14.3478,
    48: 14.5028,
    49: 14.9899,
    50: 15.0010,
    51: 15.3276,
    52: 15.5324,
    53: 15.8258,
    54: 16.1911,
    55: 16.4236,
    56: 16.6395,
    57: 16.9882,
    58: 17.1930,
    59: 17.4753,
    60: 17.7466,
    61: 18.0178,
    62: 18.2669,
    63: 18.4606,
    64: 18.8813,
    65: 18.9367,
    66: 19.3352,
    67: 19.5788,
    68: 20.0050,
    69: 20.1323,
    70: 20.5918,
    71: 20.7911,
    72: 21.1398,
    73: 21.4221,
    74: 21.8151,
    75: 21.9923,
    76: 22.4683,
    77: 22.5845,
    78: 22.7838,
    79: 23.1768,
    80: 23.3208,
    81: 23.8245,
    82: 23.9739,
    83: 24.2396,
    84: 24.5552,
    85: 25.0312,
    86: 25.2194,
    87: 25.6456,
    88: 25.8560,
    89: 26.4261,
    90: 26.5036,
    91: 26.7804,
    92: 27.1070,
    93: 27.6439,
    94: 27.8100,
    95: 28.2085,
    96: 28.4576,
    97: 28.7233,
    98: 29.1219,
    99: 29.5094,
    100: 29.8083,
    101: 30.2511,
    102: 30.5279,
    103: 30.7382,
    104: 31.0704,
    105: 31.4412,
    106: 31.7678,
    107: 32.1110,
    108: 32.6812,
    109: 32.9635,
    110: 33.5834,
    111: 34.0207,
    112: 34.3418,
    113: 34.6850,
    114: 35.0669,
    115: 35.5208,
    116: 35.6648,
    117: 36.1021,
    118: 36.2958,
    119: 36.6667,
    120: 36.7552,
    121: 37.0597,
    122: 37.1095,
    123: 37.4472,
    124: 37.7571,
    125: 37.9343,
    126: 38.2609,
    127: 38.6096,
    128: 38.7314,
}


POSE_MODE_PATH = "weights/pose_instrument_m_260825.pt"
pose_cls_names = ["instrument", "instrument_led"]
READING_KPT_CONF = 0.2

min_value = 0
max_value = 2.5
total_value = max_value - min_value
readings = [min_value, min_value + total_value * 0.2, min_value + total_value * 0.4, min_value + total_value * 0.6, min_value + total_value * 0.8, max_value]


# =========================
# PTZ zoom_step 实时监控
# =========================
class ZoomStatusMonitor:
    """通过 Dahua AttachPTZStatusProc 回调获取实时 zoom_step。"""

    def __init__(self, host: str, port: int, username: str, password: str, channel: int) -> None:
        from ctypes import POINTER, cast, sizeof
        from NetSDK.NetSDK import NetClient
        from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect, fPTZStatusProcCallBack
        from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE
        from NetSDK.SDK_Struct import (
            C_LLONG,
            NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
            NET_IN_PTZ_STATUS_PROC,
            NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
            NET_OUT_PTZ_STATUS_PROC,
            SDK_PTZ_LOCATION_INFO,
        )

        self._cast = cast
        self._sizeof = sizeof
        self._POINTER = POINTER
        self._SDK_PTZ_LOCATION_INFO = SDK_PTZ_LOCATION_INFO
        self._NET_IN_PTZ_STATUS_PROC = NET_IN_PTZ_STATUS_PROC
        self._NET_OUT_PTZ_STATUS_PROC = NET_OUT_PTZ_STATUS_PROC
        self._NET_IN_LOGIN = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY
        self._NET_OUT_LOGIN = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY
        self._EM_LOGIN_SPAC_CAP_TYPE = EM_LOGIN_SPAC_CAP_TYPE

        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.channel = channel

        self.login_id = C_LLONG()
        self.attach_handle = C_LLONG()
        self._lock = __import__("threading").Lock()
        self._zoom_step: int | None = None
        self._last_valid_status: dict | None = None

        self.sdk = NetClient()
        self._disconnect_cb = fDisConnect(self._on_disconnect)
        self._reconnect_cb = fHaveReConnect(self._on_reconnect)
        self._ptz_status_cb = fPTZStatusProcCallBack(self._on_ptz_status)
        self.sdk.InitEx(self._disconnect_cb)
        self.sdk.SetAutoReconnect(self._reconnect_cb)

    def _on_disconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser) -> None:
        print(f"[PTZ 断线] {self.host}:{self.port}")

    def _on_reconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser) -> None:
        print(f"[PTZ 重连] {self.host}:{self.port}")

    def _on_ptz_status(self, lLoginID, lAttachHandle, pBuf, nBufLen, dwUser) -> None:
        if not pBuf:
            return
        ptz_info = self._cast(pBuf, self._POINTER(self._SDK_PTZ_LOCATION_INFO)).contents
        zoom = ptz_info.nPTZZoom
        if zoom > 0:
            with self._lock:
                self._zoom_step = zoom

    def login(self) -> None:
        stu_in = self._NET_IN_LOGIN()
        stu_in.dwSize = self._sizeof(self._NET_IN_LOGIN)
        stu_in.szIP = self.host.encode()
        stu_in.nPort = self.port
        stu_in.szUserName = self.username.encode()
        stu_in.szPassword = self.password.encode()
        stu_in.emSpecCap = self._EM_LOGIN_SPAC_CAP_TYPE.TCP
        stu_in.pCapParam = None

        stu_out = self._NET_OUT_LOGIN()
        stu_out.dwSize = self._sizeof(self._NET_OUT_LOGIN)

        self.login_id, _, err = self.sdk.LoginWithHighLevelSecurity(stu_in, stu_out)
        if self.login_id == 0:
            raise RuntimeError(f"PTZ 登录失败: {err}")
        print(f"PTZ 状态监控登录成功: {self.host}:{self.port}")

    def attach(self) -> None:
        stu_in = self._NET_IN_PTZ_STATUS_PROC()
        stu_in.dwSize = self._sizeof(self._NET_IN_PTZ_STATUS_PROC)
        stu_in.nChannel = self.channel
        stu_in.cbPTZStatusProc = self._ptz_status_cb
        stu_in.dwUser = 0

        stu_out = self._NET_OUT_PTZ_STATUS_PROC()
        stu_out.dwSize = self._sizeof(self._NET_OUT_PTZ_STATUS_PROC)

        self.attach_handle = self.sdk.AttachPTZStatusProc(self.login_id, stu_in, stu_out, 5000)
        if not self.attach_handle:
            raise RuntimeError(f"AttachPTZStatusProc 失败: {self.sdk.GetLastErrorMessage()}")
        print("PTZ 状态订阅成功")

    def get_zoom(self) -> int | None:
        with self._lock:
            return self._zoom_step

    def cleanup(self) -> None:
        if self.attach_handle:
            self.sdk.DetachPTZStatusProc(self.attach_handle)
            self.attach_handle = 0
        if self.login_id:
            self.sdk.Logout(self.login_id)
            self.login_id = 0
        self.sdk.Cleanup()


@dataclass
class PosePoint:
    x: int
    y: int
    conf: float = 1.0


@dataclass
class PoseDetection:
    cls_name: str
    box: tuple[int, int, int, int]
    points: list[PosePoint]
    reading: float | None = None


class SecondaryPoseDetector:
    """Run pose estimation on cropped regions and map keypoints back to full frame."""

    def __init__(self, model_path: str | Path, target_names: list[str]) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"关键点模型文件不存在: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.target_names = set(target_names)

    @staticmethod
    def _clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = box
        x1_i = max(0, min(int(x1), width - 1))
        y1_i = max(0, min(int(y1), height - 1))
        x2_i = max(0, min(int(x2), width))
        y2_i = max(0, min(int(y2), height))
        if x2_i <= x1_i or y2_i <= y1_i:
            return None
        return x1_i, y1_i, x2_i, y2_i

    @staticmethod
    def _resolve_name(names: dict[int, str] | list[str], cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    def run_on_frame(
        self,
        frame,
        det_result,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[PoseDetection]:
        if det_result.boxes is None or len(det_result.boxes) == 0:
            return []

        h, w = frame.shape[:2]
        mapped_results: list[PoseDetection] = []

        boxes_xyxy = det_result.boxes.xyxy.tolist()
        cls_ids = det_result.boxes.cls.tolist()

        for box_xyxy, cls_id in zip(boxes_xyxy, cls_ids):
            cls_name = self._resolve_name(det_result.names, int(cls_id))
            if cls_name not in self.target_names:
                continue

            clipped = self._clip_box(box_xyxy, w, h)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pose_results = self.model.predict(
                source=crop,
                conf=conf,
                iou=iou,
                verbose=False,
            )
            if not pose_results:
                continue
            pose_result = pose_results[0]
            if pose_result.keypoints is None or pose_result.keypoints.xy is None:
                continue
            if len(pose_result.keypoints.xy) == 0:
                continue

            # 取第一条姿态结果并映射回原图坐标
            pose_xy = pose_result.keypoints.xy[0].tolist()

            points: list[PosePoint] = []
            point_confs = None
            if pose_result.keypoints.conf is not None and len(pose_result.keypoints.conf) > 0:
                point_confs = pose_result.keypoints.conf[0].tolist()

            for idx, (px, py) in enumerate(pose_xy):
                conf_v = 1.0
                if point_confs is not None and idx < len(point_confs):
                    conf_v = float(point_confs[idx])
                points.append(PosePoint(x=int(px + x1), y=int(py + y1), conf=conf_v))

            reading = self._compute_reading(points)
            mapped_results.append(
                PoseDetection(
                    cls_name=cls_name,
                    box=(x1, y1, x2, y2),
                    points=points,
                    reading=reading,
                )
            )

        return mapped_results

    @staticmethod
    def _compute_reading(points: list[PosePoint]) -> float | None:
        expected_points = len(DEFAULT_FRONT_VIEW_ANCHOR_POINTS) + 1
        if len(points) < expected_points:
            return None

        kpt_slice = points[:expected_points]
        if any(point.conf < READING_KPT_CONF for point in kpt_slice):
            return None

        src_points = [(point.x, point.y, point.conf) for point in kpt_slice]
        try:
            reading, _, _, _ = parse_readdata(src_points, readings=readings)
            return float(reading)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def draw_keypoints(frame, pose_detections: list[PoseDetection], conf_thr: float = 0.2) -> None:
        for det in pose_detections:
            for p in det.points:
                if p.conf < conf_thr:
                    continue
                cv2.circle(frame, (p.x, p.y), 3, (0, 255, 255), -1)

            if det.reading is not None:
                x1, y1, _, _ = det.box
                text = f"reading: {det.reading:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, max(30, y1 - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    (0, 255, 255),
                    4,
                    cv2.LINE_AA,
                )


@dataclass
class InferenceStats:
    total_frames: int = 0
    success_frames: int = 0
    failed_frames: int = 0
    total_boxes: int = 0


def _zoom_polling_worker(monitor: ZoomStatusMonitor, stop_event: "threading.Event", interval: float = 1.0) -> None:
    """后台线程: 每 interval 秒读取一次 PTZ zoom_step。"""
    while not stop_event.is_set():
        zoom = monitor.get_zoom()
        if zoom is not None:
            print(f"[Zoom 监控] zoom_step={zoom}")
        stop_event.wait(interval)


class RealTimeVideoDetector:
    """Run inference on a video stream and display annotated frames in real time."""

    def __init__(self, model_path: str | Path = "det_person_helmet_250821.pt") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.pose_detector = SecondaryPoseDetector(
            model_path=POSE_MODE_PATH,
            target_names=pose_cls_names,
        )

    def infer_stream(
        self,
        stream_url: str,
        rtsp_transport: str = "tcp",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: list[int] = [384, 640],
        device: str | None = None,
        window_name: str = "YOLO Real-time Detection",
        quit_key: str = "q",
        display_scale: float = 1.0,
        max_display_fps: float = 15.0,
        zoom_monitor: ZoomStatusMonitor | None = None,
    ) -> InferenceStats:
        import threading as _threading

        if rtsp_transport:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{rtsp_transport}"

        capture = LatestFrameCapture(stream_url, rtsp_transport).start()
        if not capture.cap.isOpened():
            raise RuntimeError(f"无法打开 RTSP 流: {stream_url}")

        # 启动后台 zoom 轮询线程
        zoom_stop = _threading.Event()
        zoom_thread = None
        if zoom_monitor is not None:
            zoom_thread = _threading.Thread(
                target=_zoom_polling_worker,
                args=(zoom_monitor, zoom_stop),
                daemon=True,
            )
            zoom_thread.start()

        stats = InferenceStats(total_frames=0)

        progress = TQDM(
            desc="RTSP 推理中",
            unit="帧",
        )

        frame_index = 0
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        key_code = ord(quit_key)
        window_size_inited = False
        min_frame_interval = 0.0 if max_display_fps <= 0 else 1.0 / max_display_fps
        last_show_time = 0.0

        try:
            last_frame_id = None
            while True:
                ret, frame, last_frame_id = capture.read(last_frame_id=last_frame_id, timeout=2.0)
                if not ret:
                    print("无法读取帧，流可能已断开")
                    break

                frame_index += 1

                try:
                    results = self.model.predict(
                        source=frame,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        device=device,
                        verbose=False,
                    )
                    result = results[0]
                    box_cnt = 0 if result.boxes is None else len(result.boxes)
                    stats.total_boxes += box_cnt

                    # 在原图上绘制检测框并显示，不写入任何 label 或图片文件
                    annotated_frame = result.plot(img=frame.copy())

                    # 获取当前 zoom_step 并计算 F_DIV_SENSOR
                    current_zoom = zoom_monitor.get_zoom() if zoom_monitor else None
                    f_div_sensor = F_DIV_SENSOR_BY_ZOOM.get(current_zoom) if current_zoom else None

                    # 单目测距：在检测框下方显示距离
                    if result.boxes is not None and len(result.boxes) > 0:
                        img_height = frame.shape[0]

                        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                        cls_ids = result.boxes.cls.cpu().numpy()
                        for i, box in enumerate(boxes_xyxy):
                            x1, y1, x2, y2 = box
                            box_width_px = x2 - x1
                            box_height_px = y2 - y1
                            cls_id = int(cls_ids[i]) if i < len(cls_ids) else -1
                            cls_name = result.names.get(cls_id, str(cls_id))
                            print(f"[Frame {frame_index}] {cls_name} | w={box_width_px:.0f}px h={box_height_px:.0f}px")
                            target_px = max(box_width_px, box_height_px)
                            target_size = TARGET_SIZE_MM_BY_LABEL.get(cls_name, TARGET_SIZE_MM)

                            if target_px > 0 and f_div_sensor is not None:
                                distance_mm = f_div_sensor * target_size * img_height / target_px
                                distance_m = distance_mm / 1000.0

                                text = f"{distance_m:.2f}m"
                                text_x = int(x1)
                                text_y = int(y2) + 25
                                cv2.putText(
                                    annotated_frame,
                                    text,
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2,
                                    cv2.LINE_AA,
                                )

                    # 左上角显示 zoom_step 和 F_DIV_SENSOR
                    zoom_text = f"zoom={current_zoom}" if current_zoom else "zoom=N/A"
                    fdiv_text = f"F_DIV={f_div_sensor:.4f}" if f_div_sensor else "F_DIV=N/A"
                    cv2.putText(annotated_frame, zoom_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(annotated_frame, fdiv_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                    # 二级关键点检测：仅对指定类别进行扣图并回填关键点到原图
                    pose_detections = self.pose_detector.run_on_frame(
                        frame=frame,
                        det_result=result,
                        conf=conf,
                        iou=iou,
                    )
                    self.pose_detector.draw_keypoints(annotated_frame, pose_detections)

                    if not window_size_inited:
                        show_w = max(1, int(frame.shape[1] * max(display_scale, 0.1)))
                        show_h = max(1, int(frame.shape[0] * max(display_scale, 0.1)))
                        cv2.resizeWindow(window_name, show_w, show_h)
                        window_size_inited = True

                    cv2.imshow(window_name, annotated_frame)

                    stats.success_frames += 1
                except Exception as exc:  # noqa: BLE001
                    stats.failed_frames += 1
                    print(f"[失败] 第 {frame_index} 帧: {exc}")
                finally:
                    #progress.update(1)
                    pass

                elapsed = time.perf_counter() - last_show_time
                remain = max(0.0, min_frame_interval - elapsed)
                delay_ms = max(1, int(remain * 1000)) if remain > 0 else 1

                key = cv2.waitKey(delay_ms) & 0xFF
                if key == key_code:
                    print(f"检测到退出按键: {quit_key}")
                    break
                elif key == ord("s"):
                    save_path = f"./测距存图/screenshot_{frame_index}.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"截图已保存: {save_path}")
                last_show_time = time.perf_counter()
        finally:
            zoom_stop.set()
            if zoom_thread is not None:
                zoom_thread.join(timeout=2.0)
            progress.close()
            capture.release()
            cv2.destroyAllWindows()

        return stats


def main() -> None:
    # 启动 PTZ zoom_step 监控
    zoom_monitor = ZoomStatusMonitor(
        host=PTZ_HOST, port=PTZ_PORT,
        username=PTZ_USERNAME, password=PTZ_PASSWORD,
        channel=PTZ_CHANNEL,
    )
    zoom_monitor.login()
    zoom_monitor.attach()

    detector = RealTimeVideoDetector(model_path=MODEL_PATH)

    stats = detector.infer_stream(
        stream_url=RTSP_URL,
        rtsp_transport=RTSP_TRANSPORT,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        window_name=WINDOW_NAME,
        quit_key=QUIT_KEY,
        display_scale=DISPLAY_SCALE,
        max_display_fps=MAX_DISPLAY_FPS,
        zoom_monitor=zoom_monitor,
    )

    # 结果打印
    print("="*50)
    print(f"有效推理成功帧: {stats.success_frames}")
    print(f"推理失败帧: {stats.failed_frames}")
    print(f"检测目标总框数: {stats.total_boxes}")
    print(f"RTSP 流地址: {RTSP_URL}")
    print("="*50)

    zoom_monitor.cleanup()


if __name__ == "__main__":
    main()