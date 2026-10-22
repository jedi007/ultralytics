# coding=utf-8
import argparse
import time
from ctypes import sizeof
from typing import Dict

from NetSDK.NetSDK import NetClient
from NetSDK.SDK_Callback import fDisConnect, fHaveReConnect
from NetSDK.SDK_Enum import EM_LOGIN_SPAC_CAP_TYPE, SDK_PTZ_ControlType
from NetSDK.SDK_Struct import (
	C_LLONG,
	NET_IN_LOGIN_WITH_HIGHLEVEL_SECURITY,
	NET_OUT_LOGIN_WITH_HIGHLEVEL_SECURITY,
)


COMMAND_MAP = {
	"up": SDK_PTZ_ControlType.UP_CONTROL,
	"down": SDK_PTZ_ControlType.DOWN_CONTROL,
	"left": SDK_PTZ_ControlType.LEFT_CONTROL,
	"right": SDK_PTZ_ControlType.RIGHT_CONTROL,
	"leftup": SDK_PTZ_ControlType.LEFTTOP,
	"rightup": SDK_PTZ_ControlType.RIGHTTOP,
	"leftdown": SDK_PTZ_ControlType.LEFTDOWN,
	"rightdown": SDK_PTZ_ControlType.RIGHTDOWN,
	"zoomin": SDK_PTZ_ControlType.ZOOM_ADD_CONTROL,
	"zoomout": SDK_PTZ_ControlType.ZOOM_DEC_CONTROL,
	"focusadd": SDK_PTZ_ControlType.FOCUS_ADD_CONTROL,
	"focusdec": SDK_PTZ_ControlType.FOCUS_DEC_CONTROL,
	"apertureopen": SDK_PTZ_ControlType.APERTURE_ADD_CONTROL,
	"apertureclose": SDK_PTZ_ControlType.APERTURE_DEC_CONTROL,
}

COMMAND_HELP = [
	("up/down/left/right", "云台上下左右移动"),
	("leftup/rightup/leftdown/rightdown", "云台斜向移动"),
	("zoomin/zoomout", "镜头变倍+ / 变倍-"),
	("focusadd/focusdec", "镜头对焦+ / 对焦-"),
	("apertureopen/apertureclose", "光圈+ / 光圈-"),
]


class DahuaPtzDemo:
	def __init__(self, host: str, port: int, username: str, password: str, channel: int) -> None:
		self.host = host
		self.port = port
		self.username = username
		self.password = password
		self.channel = channel
		self._active_commands: Dict[str, int] = {}

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

	def start_ptz(self, command_name: str, speed: int) -> None:
		self._validate_command(command_name, speed)
		command = COMMAND_MAP[command_name]
		self._send_command(command, speed, False)
		self._active_commands[command_name] = speed

	def stop_ptz(self, command_name: str, speed: int | None = None) -> None:
		if command_name not in COMMAND_MAP:
			raise ValueError(f"不支持的命令: {command_name}")
		if command_name not in self._active_commands:
			return
		resolved_speed = speed if speed is not None else self._active_commands.get(command_name, 4)
		if not 1 <= resolved_speed <= 8:
			raise ValueError("speed 必须在 1 到 8 之间")
		command = COMMAND_MAP[command_name]
		self._send_command(command, resolved_speed, True)
		self._active_commands.pop(command_name, None)

	def stop_active_ptz(self) -> None:
		for command_name, speed in list(self._active_commands.items()):
			self.stop_ptz(command_name, speed)

	def ptz_control(self, command_name: str, speed: int, duration: float) -> None:
		self._validate_command(command_name, speed)
		if duration <= 0:
			raise ValueError("duration 必须大于 0")

		self.start_ptz(command_name, speed)
		print(f"执行命令: {command_name}, speed={speed}, duration={duration:.2f}s")
		time.sleep(duration)
		self.stop_ptz(command_name, speed)
		print(f"命令结束: {command_name}")

	def _validate_command(self, command_name: str, speed: int) -> None:
		if command_name not in COMMAND_MAP:
			raise ValueError(f"不支持的命令: {command_name}")
		if not 1 <= speed <= 8:
			raise ValueError("speed 必须在 1 到 8 之间")

	def _send_command(self, command: SDK_PTZ_ControlType, speed: int, stop: bool) -> None:
		result = self.sdk.PTZControlEx2(
			self.login_id,
			self.channel,
			command,
			speed,
			speed,
			0,
			stop,
			None,
		)
		if not result:
			action = "停止" if stop else "开始"
			raise RuntimeError(f"{action}PTZ命令失败: {self.sdk.GetLastErrorMessage()}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="大华 NetSDK 云台转动/变倍/光圈控制 Demo。"
	)
	parser.add_argument("--host", default="192.168.5.145", help="设备 IP")
	parser.add_argument("--port", type=int, default=37777, help="设备端口")
	parser.add_argument("--username", default="admin", help="用户名")
	parser.add_argument("--password", default="sshw1234", help="密码")
	parser.add_argument("--channel", type=int, default=0, help="云台通道号")
	parser.add_argument(
		"--action",
		choices=sorted(COMMAND_MAP.keys()),
		help="单次执行的 PTZ 命令，不传则进入交互模式",
	)
	parser.add_argument("--speed", type=int, default=4, help="速度，范围 1-8")
	parser.add_argument("--duration", type=float, default=1.0, help="持续时间，单位秒")
	return parser


def print_help() -> None:
	print("可用命令:")
	for command_text, description in COMMAND_HELP:
		print(f"  {command_text} [speed] [duration]  {description}")
	print("  help")
	print("  quit")


def interactive_loop(demo: DahuaPtzDemo) -> None:
	print("进入交互模式，输入 help 查看命令。")
	while True:
		raw = input("ptz> ").strip()
		if not raw:
			continue
		if raw in {"quit", "exit", "q"}:
			break
		if raw == "help":
			print_help()
			continue

		parts = raw.split()
		command_name = parts[0].lower()
		speed = int(parts[1]) if len(parts) >= 2 else 4
		duration = float(parts[2]) if len(parts) >= 3 else 1.0

		try:
			demo.ptz_control(command_name, speed, duration)
		except Exception as exc:
			print(f"执行失败: {exc}")


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	demo = DahuaPtzDemo(
		host=args.host,
		port=args.port,
		username=args.username,
		password=args.password,
		channel=args.channel,
	)

	try:
		demo.login()
		if args.action:
			demo.ptz_control(args.action, args.speed, args.duration)
		else:
			print_help()
			interactive_loop(demo)
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
