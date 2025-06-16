import torch
import torch.nn as nn

class Tui(nn.Module):
    def __init__(self):
        super(Tui, self).__init__()
        self.m1 = torch.nn.Conv2d(3, 3, 3, 3)
        self.m2 = torch.nn.Conv2d(3, 3, 3, 3)
        # self.m2.requires_grad = False
        # for p in self.m2.parameters():
        #     p.requires_grad = False
        self.m3 = torch.nn.Conv2d(3, 3, 3, 3)

    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(x1)
        # with torch.no_grad():
        #     x2 = self.m2(x1)
        x3 = self.m3(x2)
        return x3

if __name__ == '__main__':
    
    tui = Tui().cuda()
    optim = torch.optim.SGD(tui.parameters(), lr=0.01)
    
    protype1 = torch.rand(1, 3, 3, 3).cuda()
    protype2 = torch.rand(1, 3, 3, 3).cuda()

    for epoch in range(3):
        outputs = tui(torch.rand(1, 3, 100, 100).cuda())
        
        
        # 更新一下全局变量
        # protype1 = protype1 * 0.8 + outputs * 0.2
        # loss2 = (protype1 - protype2).mean()
        # print(outputs.shape)
        loss1 = torch.mean(outputs)
        optim.zero_grad()# 梯度清零————要把网络模型当中每一个 调节 参数梯度 调为0，参数清零
        # loss = loss1 + loss2
        loss1.backward()# 反向传播求解梯度————调用存损失函数的反向传播，求出每个节点的梯度，
        print(outputs.shape, outputs.data, outputs)
        print(outputs.data.grad, outputs.grad)
        optim.step()#   更新权重参数————调用优化器，对每个参数进行调优，循环
        # print('tui.m1 ', tui.m1.weight[0, 0, 0, :])
        # print('tui.m2 ', tui.m2.weight[0, 0, 0, :])
        # print('tui.m3 ', tui.m3.weight[0, 0, 0, :])
        # print('##########')