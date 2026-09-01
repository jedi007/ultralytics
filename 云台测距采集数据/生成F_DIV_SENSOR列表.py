#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取 zoom calibration txt，按公式生成 F_DIV_SENSOR 列表。

用法: python3 生成F_DIV_SENSOR列表.py 云台测距采集数据/zoom_step_5.50m_16-128.txt
"""

import re
import sys
from pathlib import Path

# 固定参数
TARGET_SIZE_MM = 92.0   # 目标实际高度 (mm)
IMG_HEIGHT = 1080       # 图像高度 (px)


def parse_filename_distance(filepath: Path) -> float:
    """从文件名中提取距离，如 zoom_step_5.50m_16-128.txt → 5.50 (m)。"""
    match = re.search(r'(\d+\.?\d*)m', filepath.name)
    if not match:
        raise ValueError(f"文件名中未找到距离参数 (如 x.xxm): {filepath.name}")
    return float(match.group(1))


def parse_file_content(filepath: Path) -> list[tuple[int, float]]:
    """解析文件内容，返回 [(zoom_step, pixel), ...]。"""
    records = []
    for line in filepath.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        # zoom_step:16 ==> 106.2
        m = re.match(r'zoom_step:(\d+)\s*==>\s*([\d.]+)', line)
        if m:
            records.append((int(m.group(1)), float(m.group(2))))
    return records


def calc_f_div_sensor(distance_m: float, pixel: float) -> float:
    """F_DIV_SENSOR = 距离(mm) * 像素值 / (TARGET_SIZE_MM * IMG_HEIGHT)。"""
    distance_mm = distance_m * 1000.0
    return distance_mm * pixel / (TARGET_SIZE_MM * IMG_HEIGHT)


def main() -> int:
    file_path = "云台测距采集数据/zoom_step_3.773m_1-15.txt"

    filepath = Path(file_path)
    if not filepath.exists():
        print(f"文件不存在: {filepath}")
        return 1

    distance_m = parse_filename_distance(filepath)
    records = parse_file_content(filepath)

    if not records:
        print("文件内容为空或格式不正确")
        return 1

    print(f"文件: {filepath.name}")
    print(f"距离: {distance_m}m ({distance_m * 1000:.0f}mm)")
    print(f"目标尺寸: {TARGET_SIZE_MM}mm, 图像高度: {IMG_HEIGHT}px")
    print(f"公式: F_DIV_SENSOR = {distance_m * 1000:.0f} * pixel / ({TARGET_SIZE_MM} * {IMG_HEIGHT})")
    print(f"     F_DIV_SENSOR = {distance_m * 1000:.0f} * pixel / {TARGET_SIZE_MM * IMG_HEIGHT:.0f}")
    print()
    print(f"{'zoom_step':>10}  {'pixel':>8}  {'F_DIV_SENSOR':>14}")
    print("-" * 36)

    for zoom_step, pixel in records:
        f_div = calc_f_div_sensor(distance_m, pixel)
        print(f"{zoom_step:>10}  {pixel:>8.1f}  {f_div:>14.4f}")

    print()
    # 打印 Python dict 格式方便复制
    print("# Python dict 格式:")
    print("F_DIV_SENSOR_BY_ZOOM = {")
    for zoom_step, pixel in records:
        f_div = calc_f_div_sensor(distance_m, pixel)
        print(f"    {zoom_step}: {f_div:.4f},")
    print("}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
