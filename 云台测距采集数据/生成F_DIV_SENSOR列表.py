#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取插值后的 zoom calibration txt，生成 F_DIV_SENSOR 列表并输出为 .py 文件。

用法: python3 生成F_DIV_SENSOR列表.py
"""

import os
from pathlib import Path

# 固定参数
TARGET_SIZE_MM = 92.0   # 目标实际高度 (mm)
IMG_HEIGHT = 1080       # 图像高度 (px)
DISTANCE_M = 3.995      # 距离 (m)


def parse_interpolated_file(filepath: Path) -> list[tuple[int, float]]:
    """解析插值后的文件，返回 [(abs_zoom, pix), ...]。"""
    records = []
    lines = filepath.read_text(encoding='utf-8').splitlines()
    
    # 跳过第一行表头
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            abs_zoom = int(parts[0])
            pix = float(parts[1])
            records.append((abs_zoom, pix))
    
    return records


def calc_f_div_sensor(distance_m: float, pixel: float) -> float:
    """F_DIV_SENSOR = 距离(mm) * 像素值 / (TARGET_SIZE_MM * IMG_HEIGHT)。"""
    distance_mm = distance_m * 1000.0
    return distance_mm * pixel / (TARGET_SIZE_MM * IMG_HEIGHT)


def main() -> int:
    # 输入文件路径
    script_dir = Path(__file__).parent
    input_file = script_dir / 'zoom_3.995m_interpolated.txt'
    
    if not input_file.exists():
        print(f"文件不存在: {input_file}")
        return 1

    records = parse_interpolated_file(input_file)

    if not records:
        print("文件内容为空或格式不正确")
        return 1

    print(f"文件: {input_file.name}")
    print(f"距离: {DISTANCE_M}m ({DISTANCE_M * 1000:.0f}mm)")
    print(f"目标尺寸: {TARGET_SIZE_MM}mm, 图像高度: {IMG_HEIGHT}px")
    print(f"公式: F_DIV_SENSOR = {DISTANCE_M * 1000:.0f} * pixel / ({TARGET_SIZE_MM} * {IMG_HEIGHT})")
    print(f"数据条数: {len(records)}")
    print()

    # 生成字典
    f_div_dict = {}
    for abs_zoom, pix in records:
        f_div = calc_f_div_sensor(DISTANCE_M, pix)
        f_div_dict[abs_zoom] = round(f_div, 4)

    # 输出到 .py 文件
    output_file = script_dir / 'F_DIV_SENSOR_list.py'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write("# -*- coding: utf-8 -*-\n")
        f.write(f'"""F_DIV_SENSOR 列表 (距离: {DISTANCE_M}m)"""\n\n')
        f.write("F_DIV_SENSOR_BY_ZOOM = {\n")
        for abs_zoom, f_div in sorted(f_div_dict.items()):
            f.write(f"    {abs_zoom}: {f_div:.4f},\n")
        f.write("}\n")

    print(f"已生成文件: {output_file}")
    print(f"字典键值对数量: {len(f_div_dict)}")
    
    # 打印部分数据预览
    print("\n数据预览 (前10条):")
    for i, (abs_zoom, f_div) in enumerate(sorted(f_div_dict.items())[:10]):
        print(f"    {abs_zoom}: {f_div:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
