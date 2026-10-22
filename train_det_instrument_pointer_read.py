from ultralytics import YOLO


if __name__ == '__main__':  
    # 加载模型
    # model = YOLO("yolov8_personup_helmet.yaml")  # 从头开始构建新模型
    model = YOLO("yolo26s.pt")  # 加载预训练模型（建议用于训练）

    # 使用模型
    model.train(data="data_instrument_pointer_read.yaml", epochs=50, batch=32, cfg="hyp_instrument_det_pointer_read_cfg.yaml", workers = 4, device=0)  # 训练模型
    metrics = model.val()  # 在验证集上评估模型性能

    model.export(format="onnx", device=0, imgsz=(640, 640),dynamic=True, simplify=True)