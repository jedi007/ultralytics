from ultralytics import YOLO
import torch

model = YOLO("cls_smoking_20241113_e12.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, imgsz=(128, 128),dynamic=True, simplify=True)