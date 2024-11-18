from ultralytics import YOLO
model = YOLO('cls_smoking_241118_e4.pt')  # load an official model
success = model.export(format="onnx",imgsz=128,optimize =True,simplify=True, opset=11)