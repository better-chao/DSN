# 这个test.py文件主要是对mmdetection的focal loss 的原理核代码进行探究
import torch
import torch.nn.functional as F
import numpy as np

# 这个输入代表的就是我们的预测值
# 这是一个多分类任务，也就是我们的这里是一个样本，但是是一个5分类的任务
input = torch.tensor([[0.1, 0.3, 0.8, 0.1, 0.1]]).cuda(1)
print("input:", input)
pred_sigmoid = input.sigmoid()
print("pred_sigmoid", pred_sigmoid)
pytarget = torch.tensor(np.array([[0, 1, 0, 0, 0]])).cuda(1)
print("pytarget", pytarget)
pytarget = pytarget.type_as(input)
print("pytarget", pytarget)
pt = (1 - pred_sigmoid) * pytarget + pred_sigmoid * (1 - pytarget)
# pytarget = 0, pt = pred_sigmoid
# pytarget = 1, pt = 1 - pred_sigmoid
# 因为这是一个2分类任务，因此 pytarget = 0 是代表正类 ，正类比较少，用到的系数就是0.75 ，父类比较多，用到的系数就是0.25，通过这种方式来进行平衡
print("pt:", pt)
alpha = float(0.25)
gamma = float(2.0)
focal_weight = (alpha * pytarget + (1 - alpha) * (1 - pytarget)) * pt.pow(gamma)
# pytarget = 0, focal_weight = (1-alpha)*pt.pow(gamma)
# pytarget = 1, focal_weight = alpha*pt.pow(gamma)
print("focal_weight", focal_weight)
loss = F.binary_cross_entropy_with_logits(input, pytarget, reduction='none') * focal_weight
print("cross_entropy:", F.binary_cross_entropy_with_logits(input, pytarget, reduction='none'))
print("loss = ", loss)