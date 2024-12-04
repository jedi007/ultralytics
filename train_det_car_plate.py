from ultralytics import YOLO


if __name__ == '__main__':  
    # 加载模型
    # model = YOLO("yolov8_personup_helmet.yaml")  # 从头开始构建新模型
    model = YOLO("yolov8s.pt")  # 加载预训练模型（建议用于训练）

    # 使用模型
    model.train(data="det_car_plate.yaml", epochs=25, cfg="det_car_plate_super.yaml", device=0, batch=32)  # 训练模型
    # metrics = model.val()  # 在验证集上评估模型性能

    model.export(format="onnx", device=0, imgsz=(384, 640),dynamic=True, simplify=True)