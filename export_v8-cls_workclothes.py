from ultralytics import YOLO
import torch

model = YOLO("cls_workclothes_241123_s_e12.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, imgsz=(192, 192), dynamic=True, simplify=True)