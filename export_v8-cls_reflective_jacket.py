from ultralytics import YOLO
import torch

model = YOLO("cls_reflective_jacket_20241024.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, imgsz=(192, 192), dynamic=True, simplify=True)