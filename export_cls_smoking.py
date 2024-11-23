from ultralytics import YOLO
import torch

model = YOLO("cls_calling_241121_s_e17.pt")
# model.half()
# model.export(format="onnx", dynamic=True, batch=-1)
model.export(format="onnx", device=0, imgsz=(128, 128),dynamic=True, simplify=True)