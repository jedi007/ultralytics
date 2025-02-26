import sys
import os

# 获取上一级目录的绝对路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将上一级目录添加到 sys.path 中
sys.path.append(parent_dir)

from ultralytics import YOLO

model = YOLO("权重文件/release/cls_work_clothes_e16_250226.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, imgsz=(192, 192), dynamic=True, simplify=True)