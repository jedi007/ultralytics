from ultralytics import YOLO

# Load a pretrained YOLO26n model
model = YOLO("yolo26n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",  # Path to dataset configuration file
    epochs=5,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model(R"E:\code\github_code\testdata\1.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model



# 带NMS导出
from ultralytics import YOLO

# 加载YOLO26模型（n/l/m/x，根据自己的权重选择）
model = YOLO("yolo26n.pt")  # 替换为你的yolo26权重文件

# 导出带NMS的ONNX模型（核心：框架自动嵌入DIoU-NMS，无需额外配置）
success = model.export(
    format="onnx",        # 导出格式为ONNX
    opset=12,             # ONNX算子集，≥11即可，TensorRT8+推荐12
    dynamic=False,         # 支持动态输入尺寸 (1,3,height,width)
    conf=0.45,            # 嵌入NMS的置信度阈值（可后续在C++中调整）
    iou=0.8,             # 嵌入NMS的IOU阈值（可后续在C++中调整）
    max_det=300,          # 单图最大检测框数
    simplify=True         # 简化ONNX模型，去除冗余算子，提升TensorRT推理速度
)
print("ONNX模型导出成功：" if success else "ONNX模型导出失败")