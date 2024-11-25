import torch
import torch.nn as nn

# # 定义一个简单的模型
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc = nn.Linear(10, 1)

#     def forward(self, x):
#         return self.fc(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# 加载模型时，可以指定map_location来改变设备
loaded_model = torch.jit.load("scripted_model.torchscript", map_location=device)
# loaded_model = torch.jit.load("scripted_model.torchscript")

# 现在可以使用 loaded_model 来进行推理
example_input = torch.randn(1, 10, device=device)
output = loaded_model(example_input)
print(output)

