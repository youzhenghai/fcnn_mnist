import torch
import torch.nn as nn

class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(784, 10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #x = torch.softmax(x, dim=1) #  会更收敛吗？ 实际上 crossentropyloss 里面已经有了softmax  debug 看看
        return x

class NeuralNetworkClassifier(nn.Module):
    def __init__(self):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# #2.设计模型
# class Net(torch.nn.Module):
#     def __init__(self):
#     	super(Net, self).__init__()
#     	self.l1 = torch.nn.Linear(784, 512)
#     	self.l2 = torch.nn.Linear(512, 256)
#     	self.l3 = torch.nn.Linear(256, 128)
#     	self.l4 = torch.nn.Linear(128, 64)
#     	self.l5 = torch.nn.Linear(64, 10)
    
#     def forward(self, x):
#     	x = x.view(-1, 784)  
#     	x = F.relu(self.l1(x))
#     	x = F.relu(self.l2(x))
#     	x = F.relu(self.l3(x))
#     	x = F.relu(self.l4(x))
#     	return self.l5(x)   #注意最后一层不做激活
