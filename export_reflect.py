from ultralytics import YOLO
import torch

model = YOLO("det_reflect_241018.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, imgsz=(96, 256),dynamic=True, simplify=True)