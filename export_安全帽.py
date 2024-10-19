from ultralytics import YOLO
import torch

model = YOLO("跌倒安全帽安全绳V0.7.1_金宏负样本增强_20240902.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, imgsz=(384, 640),dynamic=True, simplify=True)