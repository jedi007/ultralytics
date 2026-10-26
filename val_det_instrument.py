from ultralytics import YOLO


if __name__ == '__main__':  
    # 加载模型
    # model = YOLO("yolov8_personup_helmet.yaml")  # 从头开始构建新模型
    model = YOLO("weights/det_instrument_pointer_tb_bets.pt")  # 加载预训练模型

    # 评估模型
    metrics = model.val(data="data_instrument_pointer_read.yaml", batch=32, workers=4, device=0)  # 在验证集上评估模型性能