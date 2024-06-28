from ultralytics import YOLO
import torch

model = YOLO("yolov8m.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, dynamic=True, simplify=True)