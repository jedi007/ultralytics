from ultralytics import YOLO
import torch

model = YOLO("smoking_detV0.2_20240809.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, imgsz=(224, 112), dynamic=True, simplify=True)