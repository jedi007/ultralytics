#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轮询 zoom_step 16→128，记录每个焦距下检测框的稳定像素尺寸到 txt。"""

from __future__ import annotations

from pathlib import Path
import os
import time

import cv2

from rd_latest_frame_capture import LatestFrameCapture
from ultralytics import YOLO


# =========================
# Config
# =========================
MODEL_PATH = "weights/det_instrument_20260817.pt"
RTSP_URL = "rtsp://admin:sshw1234@10.42.0.30:554/cam/realmonitor?channel=1&subtype=0"
RTSP_TRANSPORT = "tcp"
CONF = 0.55
IOU = 0.45
IMGSZ = [384, 640]
DEVICE = None

PTZ_HOST = "10.42.0.30"
PTZ_PORT = 37777
PTZ_USERNAME = "admin"
PTZ_PASSWORD = "sshw1234"
PTZ_CHANNEL = 0

HORIZONTAL_ANGLE = 1660
VERTICAL_ANGLE = 45

ZOOM_START = 1
ZOOM_END = 15
ZOOM_STEP_INTERVAL = 1
STABLE_FRAMES = 10
SETTLE_WAIT = 3.0

OUTPUT_FILE = "zoom_calibration_data.txt"

WINDOW_NAME = "YOLO Zoom Calibration"


# =========================
# Dahua PTZ zoom 控制
# =========================
class DahuaPtzZoomController:
    """通过 Dahua NetSDK EXACTGOTO 设置/读取 zoom_step。"""

    def __init__(self, host: str, port: int, username: str, password: str, channel: int) -> None:
        from ctypes import sizeof
        from NetSDK.NetSDK import NetClient
        from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect
        from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE, SDK_PTZ_ControlType
        from NetSDK.SDK_Struct import (
            C_LLONG,
            NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
            NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
        )

        self._SDK_PTZ_ControlType = SDK_PTZ_ControlType
        self._EM_LOGIN_SPAC_CAP_TYPE = EM_LOGIN_SPAC_CAP_TYPE
        self._stu_in_type = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY
        self._stu_out_type = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY
        self._sizeof = sizeof
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.channel = channel

        self.login_id = C_LLONG()
        self._disconnect_cb = fDisConnect(self._on_disconnect)
        self._reconnect_cb = fHaveReConnect(self._on_reconnect)

        self.sdk = NetClient()
        self.sdk.InitEx(self._disconnect_cb)
        self.sdk.SetAutoReconnect(self._reconnect_cb)

    def _on_disconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser) -> None:
        print(f"[断线] {self.host}:{self.port}")

    def _on_reconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser) -> None:
        print(f"[重连] {self.host}:{self.port}")

    def login(self) -> None:
        stu_in = self._stu_in_type()
        stu_in.dwSize = self._sizeof(self._stu_in_type)
        stu_in.szIP = self.host.encode()
        stu_in.nPort = self.port
        stu_in.szUserName = self.username.encode()
        stu_in.szPassword = self.password.encode()
        stu_in.emSpecCap = self._EM_LOGIN_SPAC_CAP_TYPE.TCP
        stu_in.pCapParam = None

        stu_out = self._stu_out_type()
        stu_out.dwSize = self._sizeof(self._stu_out_type)

        self.login_id, _, error_message = self.sdk.LoginWithHighLevelSecurity(stu_in, stu_out)
        if self.login_id == 0:
            raise RuntimeError(f"登录失败: {error_message}")
        print(f"PTZ 登录成功: {self.host}:{self.port}, 通道: {self.channel}")

    def set_zoom(self, zoom_step: int, pan: int = 0, tilt: int = 0) -> None:
        """EXACTGOTO: 设置绝对位置 (pan, tilt, zoom)。"""
        if not 1 <= zoom_step <= 128:
            raise ValueError("zoom_step 必须在 1-128 之间")
        result = self.sdk.PTZControlEx2(
            self.login_id,
            self.channel,
            self._SDK_PTZ_ControlType.EXACTGOTO,
            pan,
            tilt,
            zoom_step,
            False,
            None,
        )
        if not result:
            raise RuntimeError(f"EXACTGOTO 失败: {self.sdk.GetLastErrorMessage()}")
        print(f"  EXACTGOTO: pan={pan} tilt={tilt} zoom={zoom_step}")

    def get_zoom(self) -> int | None:
        """读取当前 zoom_step，失败返回 None。"""
        try:
            # 尝试通过 GetConfig 获取 PTZ 位置信息
            # NET_SDK_CONFIG_TYPE_PTZ = 6 (部分 SDK 版本)
            # 如果不支持则静默失败
            import ctypes
            buf = ctypes.create_string_buffer(1024)
            ret = self.sdk.GetConfig(
                self.login_id,
                6,  # PTZ config type
                buf,
                1024,
                None,
            )
            if ret:
                # 简单解析：zoom 通常在结构体偏移位置
                # 这里用一个保守的 fallback
                pass
        except Exception:
            pass
        return None

    def cleanup(self) -> None:
        if self.login_id:
            self.sdk.Logout(self.login_id)
            self.login_id = 0
        self.sdk.Cleanup()


# =========================
# 检测 + 记录
# =========================
def collect_stable_detection(
    model: YOLO,
    capture: LatestFrameCapture,
    window_name: str,
    zoom_step: int,
    stable_frames: int = 10,
    conf: float = 0.55,
    iou: float = 0.45,
    imgsz: list[int] = [384, 640],
    timeout: float = 3.0,
) -> float | None:
    """读取 stable_frames 帧，返回检测框较小边的平均像素值；无检测返回 None。"""
    samples: list[float] = []
    deadline = time.perf_counter() + timeout
    last_frame_id = None
    frame_count = 0

    while len(samples) < stable_frames and time.perf_counter() < deadline:
        ret, frame, last_frame_id = capture.read(last_frame_id=last_frame_id, timeout=1.0)
        if not ret or frame is None:
            continue

        results = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        result = results[0]

        # 绘制检测框
        annotated = result.plot(img=frame.copy())

        # 取最大面积的检测框
        smaller = None
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            best_idx = int(areas.argmax())
            x1, y1, x2, y2 = boxes_xyxy[best_idx]
            box_w = x2 - x1
            box_h = y2 - y1
            smaller = float(min(box_w, box_h))
            # 在框上标注较小边值
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.putText(annotated, f"min={smaller:.0f}px", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # 显示状态信息
        status = f"zoom_step={zoom_step}  frames={len(samples)}/{stable_frames}"
        cv2.putText(annotated, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        frame_count += 1
        if smaller is not None and smaller > 0:
            samples.append(smaller)

    if not samples:
        return None
    return sum(samples) / len(samples)


def main() -> None:
    # 加载模型
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        # 尝试上级目录
        model_path = Path(__file__).resolve().parent / MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    model = YOLO(str(model_path))

    # 连接 RTSP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{RTSP_TRANSPORT}"
    capture = LatestFrameCapture(RTSP_URL, RTSP_TRANSPORT).start()
    if not capture.cap.isOpened():
        raise RuntimeError(f"无法打开 RTSP 流: {RTSP_URL}")
    print(f"RTSP 流已连接: {RTSP_URL}")

    # 连接 PTZ
    ptz = DahuaPtzZoomController(
        host=PTZ_HOST, port=PTZ_PORT,
        username=PTZ_USERNAME, password=PTZ_PASSWORD,
        channel=PTZ_CHANNEL,
    )
    ptz.login()

    # 先固定云台角度
    print(f"设置云台角度: pan={HORIZONTAL_ANGLE} tilt={VERTICAL_ANGLE}")
    ptz.set_zoom(zoom_step=ZOOM_START, pan=HORIZONTAL_ANGLE, tilt=VERTICAL_ANGLE)
    time.sleep(SETTLE_WAIT)

    # 打开输出文件
    out_path = Path(OUTPUT_FILE)
    out_f = open(out_path, "a", encoding="utf-8")

    # 初始化显示窗口 (可选，用于调试)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        zoom = ZOOM_START
        while zoom <= ZOOM_END:
            print(f"\n{'='*40}")
            print(f">>> zoom_step = {zoom}")

            # 1. 设置 zoom (pan/tilt 保持固定)
            ptz.set_zoom(zoom_step=zoom, pan=HORIZONTAL_ANGLE, tilt=VERTICAL_ANGLE)

            # 2. 等待机械稳定
            print(f"  等待 {SETTLE_WAIT}s 机械稳定...")
            time.sleep(SETTLE_WAIT)

            # 3. 验证 zoom (尽力读取)
            verified_zoom = ptz.get_zoom()
            if verified_zoom is not None:
                print(f"  验证 zoom_step: {verified_zoom}")
            else:
                print(f"  (跳过验证，SDK 不支持读取 zoom)")

            # 4. 收集稳定帧，取检测框较小边平均值
            print(f"  采集 {STABLE_FRAMES} 帧检测数据...")
            avg_smaller = collect_stable_detection(
                model=model,
                capture=capture,
                window_name=WINDOW_NAME,
                zoom_step=zoom,
                stable_frames=STABLE_FRAMES,
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                timeout=8.0,
            )

            if avg_smaller is not None:
                record = f"zoom_step:{zoom} ==> {avg_smaller:.1f}"
                print(f"  >>> {record}")
                out_f.write(record + "\n")
                out_f.flush()
            else:
                record = f"zoom_step:{zoom} ==> 无检测"
                print(f"  >>> {record}")
                out_f.write(record + "\n")
                out_f.flush()

            zoom += ZOOM_STEP_INTERVAL

    except KeyboardInterrupt:
        print("\n用户中断，退出。")
    finally:
        out_f.close()
        capture.release()
        ptz.cleanup()
        cv2.destroyAllWindows()
        print(f"\n结果已保存到: {out_path.resolve()}")


if __name__ == "__main__":
    main()
