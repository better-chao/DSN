import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


def set_seed(seed):
    # 这个是程序确定性代码
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    from torch.backends import cudnn
    cudnn.benchmark = False             # if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


# 这里写读取数据的部分
batch_size = 4

# 这里设置读取数据集的准备工作

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='./dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.l1 = torch.nn.Linear(784, 512)
        # self.l2 = torch.nn.Linear(512, 256)
        # self.l3 = torch.nn.Linear(256, 128)
        # self.l4 = torch.nn.Linear(128, 64)
        # self.l5 = torch.nn.Linear(64, 10)

        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # x = x.view(-1, 784)
        # x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # return self.l5(x)

        # Flatten data from (n, 1, 28, 28) to (n, 784)
        # print(x.shape)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.pooling(self.conv2(x)))
        # print(x.shape)
        x = x.view(batch_size, -1)  # flatten
        # print(x.shape)
        x = self.fc(x)
        return x

seed = 1234
set_seed(seed)
model = Net()
# --------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
# momentum是冲量，可以从局部极值走出来尽可能找到全局最优解
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 打印网络的可学习参数

with open('model.txt', mode='w',encoding = "utf-8") as f:
    # 模型结构只打印一次
    f.writelines(str(model) + '\n')
    # 遍历模型参数并且打印
    for name, parameters in model.named_parameters():
        f.writelines(str(name) + '\n')  # 保存参数名字
        f.writelines(str(parameters.size()) + '\n')  # 保存参数形状
        f.writelines(str(parameters) + '\n')  # 保存参数数值
    print("*" * 50)
    print("模型及其模型参数保存完成")
    print("*" * 50)

# --------------------------------------------------
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        print("target:", target)
        optimizer.zero_grad()

        # forward + backward + update
        # print("inputs.device", inputs.device)
        outputs = model(inputs)
        # print(outputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            # print(outputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

