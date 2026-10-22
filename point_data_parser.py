import math
import os
import cv2
import numpy as np

DEFAULT_FRONT_VIEW_RADIUS = 256.0
DEFAULT_FRONT_VIEW_ANCHOR_POINTS = [
    (DEFAULT_FRONT_VIEW_RADIUS, DEFAULT_FRONT_VIEW_RADIUS),    # 表盘圆心
    (
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 - math.sqrt(0.5)),
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sqrt(0.5)),
    ),                                                         # 最小值点，读数0点位 左下角7点半位置
    (
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.cos(math.radians(189.0))),
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sin(math.radians(189.0))),
    ),                                                         # 读数20点位
    (
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.cos(math.radians(243.0))),
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sin(math.radians(243.0))),
    ),                                                         # 读数40点位
    (
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.cos(math.radians(297.0))),
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sin(math.radians(297.0))),
    ),                                                         # 读数60点位
    (
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.cos(math.radians(351.0))),
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sin(math.radians(351.0))),
    ),                                                         # 读数80点位
    (
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sqrt(0.5)),
        DEFAULT_FRONT_VIEW_RADIUS * (1.0 + math.sqrt(0.5)),
    ),                                                         # 最大值点，读数100点位 在右下角4点半位置
]
READINGS = [0, 0, 20, 40, 60, 80, 100]

def parse_readdata(src_points, anchors = DEFAULT_FRONT_VIEW_ANCHOR_POINTS, readings=READINGS):
    """解析最终读数
    输入:
        src_points: 映射前的关键点列表 (x, y)或(x, y, v)，index0=圆心, index1=needle, index2+=刻度点
        anchors:    index0=圆心, index1=针尖点位,index2+=各刻度点
        readings:   与anchors一一对应的读数值列表
    输出:
        reading:        计算出的读数
        H:              单应矩阵
        needle_front:   needle映射到正视图后的坐标 (x, y)
    """
    def xy(p):
        return (float(p[0]), float(p[1]))

    src_all = [xy(p) for p in src_points]
    anchors_np = np.array([xy(a) for a in anchors], dtype=np.float32)
    readings_np = np.array(readings, dtype=np.float64)

    # 单应矩阵用圆心+刻度点(排除index1的needle)计算
    src_pts = np.array(src_all[0:1] + src_all[2:], dtype=np.float32)
    H, mask = cv2.findHomography(src_pts, anchors_np, cv2.RANSAC, 5.0)

    # needle映射到正视图
    needle_front = cv2.perspectiveTransform(
        np.array([src_all[1]], dtype=np.float32).reshape(-1, 1, 2), H
    ).reshape(2)
    center = anchors_np[0]

    def clockwise_angle(pt):
        return (math.degrees(math.atan2(pt[1] - center[1], pt[0] - center[0])) + 360.0) % 360.0

    # 计算index 1的关键点在映射后的图像中与anchors[1]点位的夹角，夹角顶点是圆心anchors[0]
    kpt1_pt = needle_front
    anchor1_pt = np.array(anchors[1], dtype=np.float64)

    angle_kpt1 = clockwise_angle(kpt1_pt)
    angle_anchor1 = clockwise_angle(anchor1_pt)
    # 读数顺时针递增，顺时针夹角 = (目标角度 - 参考角度) mod 360
    angle_between = (angle_kpt1 - angle_anchor1) % 360.0
    # print(f"index1映射点位角: {angle_kpt1:.2f}°, 0读数点角: {angle_anchor1:.2f}°, 顺时针夹角: {angle_between:.2f}°")
    
    # 根据角度计算读数
    readdata = angle_between/270.0 * (readings_np[-1] - readings_np[0]) + readings_np[0]
    # print(f"计算读数: {readdata:.2f}")

    return float(readdata), H, mask, (float(needle_front[0]), float(needle_front[1]))