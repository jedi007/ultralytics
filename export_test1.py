from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.half()
# model.export(format="onnx", dynamic=True)
model.export(format="onnx", device=0)