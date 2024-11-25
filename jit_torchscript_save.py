import torch
import torch.nn as nn

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 实例化模型并移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
model = SimpleModel().to(device)

# 如果是脚本化模型
scripted_model = torch.jit.script(model)

# 如果是跟踪模型，提供一个示例输入
example_input = torch.randn(1, 10, device=device, dtype=torch.float32)
traced_model = torch.jit.trace(model, example_input)

# 保存模型
torch.jit.save(scripted_model, "scripted_model.torchscript")
# 或者如果你选择跟踪
# torch.jit.save(traced_model, "traced_model.pt")

# 加载模型时，可以指定map_location来改变设备
# loaded_model = torch.jit.load("scripted_model.pt", map_location=torch.device('cpu'))
