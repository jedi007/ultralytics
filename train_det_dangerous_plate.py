from ultralytics import YOLO


if __name__ == '__main__':  
    # 加载模型
    # model = YOLO("yolov8_personup_helmet.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）
    model = YOLO("yolov8_dangerous_car_plate.yaml")

    # 使用模型
    model.train(data="det_car_plate.yaml", epochs=60, cfg="det_car_plate_super.yaml", device=0, batch=64, workers=6, close_mosaic=10)  # 训练模型
    # metrics = model.val()  # 在验证集上评估模型性能

    model.export(format="onnx", device=0, imgsz=(384, 640),dynamic=True, simplify=True)