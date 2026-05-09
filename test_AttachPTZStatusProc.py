# coding=utf-8
import time
from ctypes import POINTER, cast, sizeof
from datetime import datetime

from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect, fPTZStatusProcCallBack
from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE, SDK_PTZ_ControlType
from NetSDK.SDK_Struct import (
	C_LLONG,
	NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
	NET_IN_PTZ_STATUS_PROC,
	NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
	NET_OUT_PTZ_STATUS_PROC,
	SDK_PTZ_LOCATION_INFO,
)


HOST = "192.168.5.145"
PORT = 37777
USERNAME = "admin"
PASSWORD = "sshw1234"
CHANNEL = 0

MONITOR_FOREVER = True
SUBSCRIBE_SECONDS = 10

# 是否在订阅成功后主动触发一次云台移动，便于观察回调变化。
TRIGGER_EXACT_GOTO = False
TRIGGER_HORIZONTAL_ANGLE = 1000
TRIGGER_VERTICAL_ANGLE = 0
TRIGGER_ZOOM_STEP = 1


ACTION_MAP = {
	0: "Preset",
	1: "LineScan",
	2: "Cruise",
	3: "Pattern",
	4: "HorizontalRotation",
	5: "GeneralMove",
	6: "PatternRecord",
	7: "WideViewScan",
	8: "HeatMap",
	9: "AbsoluteMove",
	10: "CheckDeviceOffset",
	11: "IntelliConfigure",
	12: "Restart",
	255: "Unknown",
}

STATE_MAP = {
	0: "Unknown",
	1: "Moving",
	2: "Idle",
}


class DahuaAttachPTZStatusDemo:
	def __init__(self) -> None:
		self.login_id = C_LLONG()
		self.attach_handle = C_LLONG()
		self.sdk = NetClient()
		self._last_status = None
		self._last_valid_status = None
		self._disconnect_cb = fDisConnect(self._on_disconnect)
		self._reconnect_cb = fHaveReConnect(self._on_reconnect)
		self._ptz_status_cb = fPTZStatusProcCallBack(self._on_ptz_status)

		self.sdk.InitEx(self._disconnect_cb)
		self.sdk.SetAutoReconnect(self._reconnect_cb)

	def _on_disconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser) -> None:
		print(f"[断线] {HOST}:{PORT}")

	def _on_reconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser) -> None:
		print(f"[重连] {HOST}:{PORT}")

	def _on_ptz_status(self, lLoginID, lAttachHandle, pBuf, nBufLen, dwUser) -> None:
		if not pBuf:
			print("[PTZ回调] pBuf 为空")
			return

		ptz_info = cast(pBuf, POINTER(SDK_PTZ_LOCATION_INFO)).contents
		abs_position = ptz_info.stuAbsPosition
		status = {
			"channel": ptz_info.nChannelID,
			"pan": ptz_info.nPTZPan,
			"tilt": ptz_info.nPTZTilt,
			"zoom_step": ptz_info.nPTZZoom,
			"state": ptz_info.bState,
			"action": ptz_info.bAction,
			"abs_pan": abs_position.nPosX,
			"abs_tilt": abs_position.nPosY,
			"abs_zoom": abs_position.nZoom,
		}

		if self._should_ignore_zero_idle_status(status):
			return

		if status == self._last_status:
			return

		self._last_status = status
		if not self._is_zero_status(status):
			self._last_valid_status = status
		self._print_monitor_status(status, ptz_info.dwUTC)

	def _is_zero_status(self, status: dict) -> bool:
		return (
			status["pan"] == 0
			and status["tilt"] == 0
			and status["zoom_step"] == 0
			and status["abs_pan"] == 0
			and status["abs_tilt"] == 0
			and status["abs_zoom"] == 0
		)

	def _should_ignore_zero_idle_status(self, status: dict) -> bool:
		if status["state"] != 2:
			return False
		if not self._is_zero_status(status):
			return False
		if self._last_valid_status is None:
			return False
		return not self._is_zero_status(self._last_valid_status)

	def _print_monitor_status(self, status: dict, dw_utc: int) -> None:
		utc_text = datetime.fromtimestamp(dw_utc).strftime("%Y-%m-%d %H:%M:%S") if dw_utc else "local"
		print(
			f"[{utc_text}] "
			f"channel={status['channel']} | "
			f"pan={status['pan'] / 10:.1f}° | "
			f"tilt={status['tilt'] / 10:.1f}° | "
			f"zoom_step={status['zoom_step']} | "
			f"abs_pan={status['abs_pan'] / 100:.2f}° | "
			f"abs_tilt={status['abs_tilt'] / 100:.2f}° | "
			f"abs_zoom={status['abs_zoom']} | "
			f"state={STATE_MAP.get(status['state'], 'Unknown')} | "
			f"action={ACTION_MAP.get(status['action'], 'Unknown')}"
		)

	def login(self) -> None:
		stu_in = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
		stu_in.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
		stu_in.szIP = HOST.encode()
		stu_in.nPort = PORT
		stu_in.szUserName = USERNAME.encode()
		stu_in.szPassword = PASSWORD.encode()
		stu_in.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
		stu_in.pCapParam = None

		stu_out = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
		stu_out.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)

		self.login_id, _, error_message = self.sdk.LoginWithHighLevelSecurity(stu_in, stu_out)
		if self.login_id == 0:
			raise RuntimeError(f"登录失败: {error_message}")

		print(f"登录成功: {HOST}:{PORT}, channel={CHANNEL}")

	def attach(self) -> None:
		stu_in = NET_IN_PTZ_STATUS_PROC()
		stu_in.dwSize = sizeof(NET_IN_PTZ_STATUS_PROC)
		stu_in.nChannel = CHANNEL
		stu_in.cbPTZStatusProc = self._ptz_status_cb
		stu_in.dwUser = 0

		stu_out = NET_OUT_PTZ_STATUS_PROC()
		stu_out.dwSize = sizeof(NET_OUT_PTZ_STATUS_PROC)

		self.attach_handle = self.sdk.AttachPTZStatusProc(self.login_id, stu_in, stu_out, 5000)
		if not self.attach_handle:
			raise RuntimeError(f"AttachPTZStatusProc 失败: {self.sdk.GetLastErrorMessage()}")

		print(f"AttachPTZStatusProc 成功, attach_handle={int(self.attach_handle)}")

	def detach(self) -> None:
		if self.attach_handle:
			self.sdk.DetachPTZStatusProc(self.attach_handle)
			self.attach_handle = 0

	def trigger_exact_goto(self) -> None:
		result = self.sdk.PTZControlEx2(
			self.login_id,
			CHANNEL,
			SDK_PTZ_ControlType.EXACTGOTO,
			TRIGGER_HORIZONTAL_ANGLE,
			TRIGGER_VERTICAL_ANGLE,
			TRIGGER_ZOOM_STEP,
			False,
			None,
		)
		if not result:
			raise RuntimeError(f"EXACTGOTO 触发失败: {self.sdk.GetLastErrorMessage()}")

		print(
			"已触发 EXACTGOTO: "
			f"pan={TRIGGER_HORIZONTAL_ANGLE / 10:.1f}°, "
			f"tilt={TRIGGER_VERTICAL_ANGLE / 10:.1f}°, "
			f"zoom_step={TRIGGER_ZOOM_STEP}"
		)

	def logout(self) -> None:
		if self.login_id:
			self.sdk.Logout(self.login_id)
			self.login_id = 0

	def cleanup(self) -> None:
		self.detach()
		self.logout()
		self.sdk.Cleanup()


def main() -> int:
	demo = DahuaAttachPTZStatusDemo()
	try:
		demo.login()
		demo.attach()
		if TRIGGER_EXACT_GOTO:
			time.sleep(1)
			demo.trigger_exact_goto()

		if MONITOR_FOREVER:
			print("开始持续监听 PTZ 当前角度，按 Ctrl+C 退出。")
			while True:
				time.sleep(1)
		else:
			print(f"开始监听 PTZ 当前角度，等待 {SUBSCRIBE_SECONDS} 秒...")
			time.sleep(SUBSCRIBE_SECONDS)
		return 0
	except KeyboardInterrupt:
		print("\n用户中断，准备退出。")
		return 130
	except Exception as exc:
		print(exc)
		return 1
	finally:
		demo.cleanup()


if __name__ == "__main__":
	raise SystemExit(main())
