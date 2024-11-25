# from ultralytics import YOLO
# import torch

# model = YOLO("cls_workclothes_241123_s_e12.pt")
# # model.half()
# # model.export(format="onnx", dynamic=True, batch=-1)
# model.export(format="torchscript", device=0, imgsz=(192, 192), dynamic=False, simplify=True)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
 
# 假设你已经加载了模型和数据
model = torch.load('cls_workclothes_241123_s_e12.pt')["ema"]  # 加载模型
model.eval()  # 设置为评估模式
model.to(device)
 
example_input = torch.rand(1, 3, 192, 192, device=device, dtype=torch.float16)  # 创建一个示例输入
 
# 使用torch.jit.trace来创建一个静态模型
static_model = torch.jit.trace(model, example_input)
 
# 保存静态模型
torch.jit.save(static_model, 'static_cls_workclothes_241123_s_e12.torchscript')
