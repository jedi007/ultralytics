from ultralytics import YOLO
import torch

model = YOLO("det_sleep_241016.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, imgsz=(384, 640),dynamic=True, simplify=True)