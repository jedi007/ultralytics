from ultralytics import YOLO


if __name__ == '__main__':  
    # 加载模型
    model = YOLO("yolov8_hat.yaml")  # 从头开始构建新模型
    # model = YOLO("yolov8n.pt")  # 加载预训练模型（建议用于训练）

    # 使用模型
    model.train(data="hat.yaml", epochs=50)  # 训练模型
    # metrics = model.val()  # 在验证集上评估模型性能