# coding=utf-8
from ctypes import sizeof

from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect
from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE, SDK_PTZ_ControlType
from NetSDK.SDK_Struct import (
	C_LLONG,
	NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
	NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
)


HOST = "192.168.5.145"
PORT = 37777
USERNAME = "admin"
PASSWORD = "sshw1234"
CHANNEL = 0

# EXACTGOTO 参数。
# horizontal_angle: 0-3600，单位 0.1 度。
# vertical_angle: -900 到 900，单位 0.1 度。
# zoom_step: 1-128，为变倍档位，不是实际倍数。
HORIZONTAL_ANGLE = 2900
VERTICAL_ANGLE = -200
ZOOM_STEP = 1


class DahuaExactGotoDemo:
	def __init__(self, host: str, port: int, username: str, password: str, channel: int) -> None:
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
		print(f"[断线] 设备连接断开: {self.host}:{self.port}")

	def _on_reconnect(self, lLoginID, pchDVRIP, nDVRPort, dwUser) -> None:
		print(f"[重连] 设备重新连接成功: {self.host}:{self.port}")

	def login(self) -> None:
		stu_in = NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY()
		stu_in.dwSize = sizeof(NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY)
		stu_in.szIP = self.host.encode()
		stu_in.nPort = self.port
		stu_in.szUserName = self.username.encode()
		stu_in.szPassword = self.password.encode()
		stu_in.emSpecCap = EM_LOGIN_SPAC_CAP_TYPE.TCP
		stu_in.pCapParam = None

		stu_out = NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY()
		stu_out.dwSize = sizeof(NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY)

		self.login_id, _, error_message = self.sdk.LoginWithHighLevelSecurity(stu_in, stu_out)
		if self.login_id == 0:
			raise RuntimeError(f"登录失败: {error_message}")

		print(f"登录成功: {self.host}:{self.port}, 通道: {self.channel}")

	def logout(self) -> None:
		if self.login_id:
			self.sdk.Logout(self.login_id)
			self.login_id = 0

	def cleanup(self) -> None:
		self.logout()
		self.sdk.Cleanup()

	def exact_goto(self, horizontal_angle: int, vertical_angle: int, zoom_step: int) -> None:
		self._validate_exact_goto(horizontal_angle, vertical_angle, zoom_step)
		result = self.sdk.PTZControlEx2(
			self.login_id,
			self.channel,
			SDK_PTZ_ControlType.EXACTGOTO,
			horizontal_angle,
			vertical_angle,
			zoom_step,
			False,
			None,
		)
		if not result:
			raise RuntimeError(f"EXACTGOTO 调用失败: {self.sdk.GetLastErrorMessage()}")

		print(
			"EXACTGOTO 已发送: "
			f"horizontal_angle={horizontal_angle / 10:.1f}°, "
			f"vertical_angle={vertical_angle / 10:.1f}°, "
			f"zoom_step={zoom_step}"
		)

	@staticmethod
	def _validate_exact_goto(horizontal_angle: int, vertical_angle: int, zoom_step: int) -> None:
		if not 0 <= horizontal_angle <= 3600:
			raise ValueError("horizontal_angle 必须在 0 到 3600 之间，单位为 0.1 度")
		if not -900 <= vertical_angle <= 900:
			raise ValueError("vertical_angle 必须在 -900 到 900 之间，单位为 0.1 度")
		if not 1 <= zoom_step <= 128:
			raise ValueError("zoom_step 必须在 1 到 128 之间")
def main() -> int:
	demo = DahuaExactGotoDemo(
		host=HOST,
		port=PORT,
		username=USERNAME,
		password=PASSWORD,
		channel=CHANNEL,
	)

	try:
		demo.login()
		demo.exact_goto(
			horizontal_angle=HORIZONTAL_ANGLE,
			vertical_angle=VERTICAL_ANGLE,
			zoom_step=ZOOM_STEP,
		)
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
