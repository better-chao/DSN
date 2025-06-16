from typing import Any, Optional, Tuple
from torch.autograd import Function
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

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

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff * 0.5, None

class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


# 这个是分类器-->实际过程中该网络应该换成自己的网络
class NormalClassifier(nn.Module):

    def __init__(self, num_features, num_classes, use_GRL=0):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
        if use_GRL:
            self.grl = GRL()

    def forward(self, x):
        if getattr(self, 'grl', None) is not None:
            x = self.grl(x)                # 注意这里
        return self.linear(x)



if __name__ == '__main__':
    # 这里就是设置了种子
    seed = 1234
    set_seed(seed)

    use_grl = 0

    net1 = NormalClassifier(3, 6)
    if use_grl == 0:
        net2 = NormalClassifier(6, 10, use_GRL = 0)           # 不使用反转层
    else:
        net2 = NormalClassifier(6, 10, use_GRL = 1)           # 使用反转层
    net3 = NormalClassifier(10, 2)

    data = torch.rand((4, 3))
    label = torch.ones((4), dtype=torch.long)
    out = net3(net2(net1(data)))
    loss = F.cross_entropy(out, label)
    loss.backward()

    
    if use_grl == 0:
        # 将梯度保存到txt文本中
        with open('nogrl.txt', mode='w',encoding = "utf-8") as f:
            f.writelines(str(net1.linear.weight.grad) + '\n')  # 保存参数名字
            f.writelines(str(net2.linear.weight.grad) + '\n')  # 保存参数形状
            f.writelines(str(net3.linear.weight.grad) + '\n')  # 保存参数形状
            print("*" * 50)
            print("nogrl文本保存完毕！")
            print("*" * 50)

    if use_grl == 1:
        # 将梯度保存到txt文本中
        with open('grl.txt', mode='w',encoding = "utf-8") as f:
            f.writelines(str(net1.linear.weight.grad) + '\n')  # 保存参数名字
            f.writelines(str(net2.linear.weight.grad) + '\n')  # 保存参数形状
            f.writelines(str(net3.linear.weight.grad) + '\n')  # 保存参数形状
            print("*" * 50)
            print("grl文本保存完毕！")
            print("*" * 50)

