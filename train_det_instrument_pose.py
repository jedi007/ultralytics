from ultralytics import YOLO
import torch

# ===================== 可配置参数（统一管理，方便修改）=====================
PRETRAINED_WEIGHTS = "yolo26m-pose.pt"  # 预训练权重
DATA_CONFIG = "data_instrument_pose.yaml"    # 数据集配置
HYP_CONFIG = "hyp_instrument_pose_cfg.yaml"   # 超参数配置
EPOCHS = 200                                  # 训练轮数
BATCH_SIZE = 64                                # 批次大小
IMAGE_SIZE = 512                              # 输入尺寸（训练/导出统一）
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # 自动判断GPU/CPU
# =========================================================================

if __name__ == '__main__':  
    # 1. 加载预训练模型（姿态估计任务必须用预训练权重初始化）
    model = YOLO(PRETRAINED_WEIGHTS)

    # 2. 训练模型（核心优化：自动保存best.pt、早停、统一尺寸、指定路径）
    print(f"开始训练，使用设备: {DEVICE}")
    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,        # 训练尺寸与导出统一
        cfg=HYP_CONFIG,
        workers=4 if DEVICE != "cpu" else 0,  # Windows/CPU自动设为0
        device=DEVICE,
        save=True,               # 开启保存（默认开启，显式声明更清晰）
        # amp=True,                # 混合精度训练，加速+省显存
        verbose=True,            # 打印详细训练日志
        save_period=1            # 🔥 核心：每1轮保存一次权重
    )  
    
    metrics = model.val()  # 在验证集上评估模型性能
    
    model.export(format="onnx", device=0, imgsz=(IMAGE_SIZE, IMAGE_SIZE),dynamic=True, simplify=True)