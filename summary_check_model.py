import torch

from torchsummary import summary
from models import SoftmaxClassifier, NeuralNetworkClassifier


# # 1. 定义与.pth文件中模型相同的模型结构
# # 2. 创建模型实例并加载.pth文件的参数
# model = NeuralNetworkClassifier()
# model.load_state_dict(torch.load('/home/shiyinglocal/recode/ml_course/neural_network_model.pth'))

# # # 3. 打印模型的摘要
# print(model)


model = NeuralNetworkClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU可用
model.load_state_dict(torch.load('/home/shiyinglocal/project/speakerbeam/example/model.pth'))
model = model.to(device)  # 将模型移到GPU或CPU上
print(model)
# 使用torchsummary查看模型摘要
summary(model, (1, 28, 28))
