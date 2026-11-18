from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent  # 脚本所在目录（train_job/）


if __name__ == '__main__':  
    # 加载模型
    # model = YOLO("yolov8_personup_helmet.yaml")  # 从头开始构建新模型
    model = YOLO("yolo26s.pt")  # 加载预训练模型（建议用于训练）

    # 使用模型
    model.train(
        data=str(ROOT / "yaml_confs/data_det_switch_light.yaml"),
        epochs=600,
        batch=32,
        cfg=str(ROOT / "yaml_confs/hyp_det_switch_light.yaml"),
        workers=4,
        device=0,
        project=str(ROOT / "runs"),   # 绝对路径
        name="switch_light",
        exist_ok=True,                 # True=复用；False=自动 switch_light2/3
    )
        
    metrics = model.val(  # 在验证集上评估模型性能
        data=str(ROOT / "yaml_confs/data_det_switch_light.yaml"),
        project=str(ROOT / "runs"),
        name="switch_light_val",
        exist_ok=True,
    )

    model.export(format="onnx", device=0, imgsz=(768, 1280),dynamic=True, simplify=True)