from ultralytics import YOLO


if __name__ == '__main__':  
    # 加载模型
    # model = YOLO("yolov8s-cls.yaml")  # 从头开始构建新模型
    model = YOLO("yolov8s-cls.pt")  # 加载预训练模型（建议用于训练）

    # 修改配置文件中的类别
    # model.nc = len(model.names)  # 设置类别数量
    # model.nc = 2

    # 使用模型
    model.train(data="/home/hyzh/DATA/cls_calling_data", epochs=30, imgsz=(128, 128), cfg="classify-calling_super.yaml")  # 训练模型
    metrics = model.val()  # 在验证集上评估模型性能
    # results = model("../datasets/test_image")  # 对图像进行预测
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式

    model.export(format="onnx", device=0, imgsz=(128, 128),dynamic=True, simplify=True)