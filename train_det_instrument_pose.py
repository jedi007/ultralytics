from ultralytics import YOLO
import torch

# ===================== 可配置参数（统一管理，方便修改）=====================
PRETRAINED_WEIGHTS = "yolo26s-pose.pt"  # 预训练权重
DATA_CONFIG = "data_instrument_pose.yaml"    # 数据集配置
HYP_CONFIG = "hyp_instrument_pose_cfg.yaml"   # 超参数配置
EPOCHS = 300                                  # 训练轮数
BATCH_SIZE = 8                                # 批次大小
IMAGE_SIZE = 256                              # 输入尺寸（训练/导出统一）
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # 自动判断GPU/CPU
PROJECT = "runs/pose"                         # 保存根目录
NAME = "instrument_pose_train"                # 任务文件夹名
PATIENCE = 50                                 # 早停：50轮无提升自动停止
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
        project=PROJECT,
        name=NAME,
        patience=PATIENCE,       # 早停，防止过拟合
        save=True,               # 开启保存（默认开启，显式声明更清晰）
        save_best=True,          # ✅ 强制保存最优模型（默认开启）
        amp=True,                # 混合精度训练，加速+省显存
        verbose=True,            # 打印详细训练日志
        save_period=1            # 🔥 核心：每1轮保存一次权重
    )  

    # 3. 验证最优模型（直接加载best.pt评估，确保用的是最高mAP模型）
    print("\n开始验证最优模型（best.pt）...")
    best_model = YOLO(f"{PROJECT}/{NAME}/weights/best.pt")
    metrics = best_model.val()  # 评估最优模型性能
    print(f"最优模型 Pose mAP50-95: {metrics.box.map50-95:.4f}")
    print(f"metrics: {metrics}")

    # 4. 导出ONNX（保持你的配置不变）
    print("\n开始导出ONNX模型...")
    best_model.export(
        format="onnx",
        device=DEVICE,
        imgsz=(IMAGE_SIZE, IMAGE_SIZE),
        dynamic=True,    # 动态batch
        simplify=True    # 简化模型
    )

    print(f"\n训练完成！最优模型路径：\n{PROJECT}/{NAME}/weights/best.pt")