from turtle import width
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from grl import set_seed, GradientReverseFunction, GRL
from utils import *
from datasets import Mnist_and_MnistM_Dataset, calc_mean
import os
from matplotlib import pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#使用编号为1的显卡
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device =", device)
import logging
# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',  # 添加时间戳
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("training.log"),  # 保存到文件
        logging.StreamHandler()  # 输出到终端
    ]
)


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

class Shared_Encode(torch.nn.Module):
    def __init__(self):
        super(Shared_Encode, self).__init__() 

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(5, 5), padding=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pooling1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pooling2 = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(3136, 100)
        self.bn3 = torch.nn.BatchNorm1d(100)
    def forward(self, x):
        
        batch_size = x.size(0)
        # 这里处理的顺序是 input--> conv-->batchnorm-->relu-->maxpooling
        x = F.relu(self.bn1(self.conv1(x)))
        # print("x.shape", x.shape)
        x = self.pooling1(x)
        # print("x.shape", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print("x.shape", x.shape)
        x = self.pooling2(x)
        # print("x.shape", x.shape)
        x = x.view(batch_size, -1)  # flatten
        # print("经过拉直之后x的shape:", x.shape)
        x = F.relu(self.bn3(self.fc(x)))
        # print("x.shape", x.shape)
        return x

class Private_Target_Encoder(torch.nn.Module):
    def __init__(self):
        super(Private_Target_Encoder, self).__init__() 

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(5, 5), padding=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pooling1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pooling2 = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(3136, 100)
        self.bn3 = torch.nn.BatchNorm1d(100)
    def forward(self, x):
        
        batch_size = x.size(0)
        # 这里处理的顺序是 input--> conv-->batchnorm-->relu-->maxpooling
        x = F.relu(self.bn1(self.conv1(x)))
        # print("x.shape", x.shape)
        x = self.pooling1(x)
        # print("x.shape", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print("x.shape", x.shape)
        x = self.pooling2(x)
        # print("x.shape", x.shape)
        x = x.view(batch_size, -1)  # flatten
        # print("经过拉直之后x的shape:", x.shape)
        x = F.relu(self.bn3(self.fc(x)))
        # print("x.shape", x.shape)
        return x

class Private_Source_Encoder(torch.nn.Module):
    def __init__(self):
        super(Private_Source_Encoder, self).__init__() 

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(5, 5), padding=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pooling1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pooling2 = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(3136, 100)
        self.bn3 = torch.nn.BatchNorm1d(100)
    def forward(self, x):
        
        batch_size = x.size(0)
        # 这里处理的顺序是 input--> conv-->batchnorm-->relu-->maxpooling
        x = F.relu(self.bn1(self.conv1(x)))
        # print("x.shape", x.shape)
        x = self.pooling1(x)
        # print("x.shape", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print("x.shape", x.shape)
        x = self.pooling2(x)
        # print("x.shape", x.shape)
        x = x.view(batch_size, -1)  # flatten
        # print("经过拉直之后x的shape:", x.shape)
        x = F.relu(self.bn3(self.fc(x)))
        # print("x.shape", x.shape)
        return x

class Shared_Decoder(torch.nn.Module):
    def __init__(self, height, width, channel):
        super(Shared_Decoder, self).__init__() 
        self.height = height
        self.width = width
        self.channel = channel

        self.fc1 = torch.nn.Linear(100, 600)
        self.bn1 = torch.nn.BatchNorm1d(600)

        self.conv1 = torch.nn.Conv2d(6, 32, kernel_size=(5, 5), padding=2)
        self.bn2 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2)
        self.bn3 = torch.nn.BatchNorm2d(32)

        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2)
        self.bn4 = torch.nn.BatchNorm2d(32)

        self.conv4 = torch.nn.Conv2d(32, self.channel, kernel_size=(5, 5), padding=2)
        self.bn5 = torch.nn.BatchNorm2d(self.channel)
        
    def forward(self, x):
        batch_size = x.size(0)
        # print("x.shape", x.shape) # 20,100
        # 这里处理的顺序是 input--> conv-->batchnorm-->relu
        x = F.relu(self.bn1(self.fc1(x)))
        # print("x.shape", x.shape) # 20,600
        x = x.view(batch_size, 6, 10, 10)
        # print("x.shape", x.shape) # 20, 6,10, 10
        x = F.relu(self.bn2(self.conv1(x)))
        # print("x.shape", x.shape) # 20, 32,10,10
        x = F.interpolate(x, size =(16, 16), mode ='nearest')
        # print("x.shape", x.shape) # 20,32,16,16
        x = F.relu(self.bn3(self.conv2(x)))
        # print("x.shape", x.shape) # 20,32,16,16
        x = F.interpolate(x, size =(32, 32), mode ='nearest')
        # print("x.shape", x.shape) # 20,32,32,32
        x = F.relu(self.bn4(self.conv3(x)))
        # print("x.shape", x.shape) # 20,32,32,32
        x = F.interpolate(x, size =(self.height, self.width), mode ='nearest')
        # print("x.shape", x.shape) # 20,32,28,28
        x = self.bn5(self.conv4(x))
        # print("x.shape", x.shape) # 20,3,28,28
        
        return x

class Classfication_Head(torch.nn.Module):
    def __init__(self):
        super(Classfication_Head, self).__init__() 
        self.fc1 = torch.nn.Linear(100, 100)
        self.fc2 = torch.nn.Linear(100, 10)
        
    def forward(self, x):
    
        x = F.relu(self.fc1(x))
        # print("x.shape", x.shape)
        x = self.fc2(x)
        # print("x.shape", x.shape)
        # 这里返回的是未做softmax的一个logits，这是因为一般softmax都是包含在损失函数中的
        return x

class Domain_Classfication(torch.nn.Module):
    def __init__(self):
        super(Domain_Classfication, self).__init__() 
        self.fc1 = torch.nn.Linear(100, 500)
        self.fc2 = torch.nn.Linear(500, 1000)
        self.fc3 = torch.nn.Linear(1000, 2000)
        self.fc4 = torch.nn.Linear(2000, 100)
        self.fc5 = torch.nn.Linear(100, 2)
        
    def forward(self, x):
    
        x = F.relu(self.fc1(x))
        # print("x.shape", x.shape)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # print("x.shape", x.shape)
        # 这里返回的是未做softmax的一个logits，这是因为一般softmax都是包含在损失函数中的
        return x

class DSN(torch.nn.Module):
    def __init__(self, shared_encoder, private_source_encoder, private_target_encoder, shared_decoder, classfication_head, domain_classfication, use_GRL=0):
        # 这里用于构建整体的网络
        super(DSN, self).__init__() 

        self.shared_encoder = shared_encoder
        self.private_source_encoder = private_source_encoder
        self.private_target_encoder = private_target_encoder
        self.shared_decoder = shared_decoder
        self.classfication_head = classfication_head
        self.domain_classfication = domain_classfication
        if use_GRL:
            self.grl = GRL()
        self.shared_s = None
        self.shared_t = None
        self.private_s = None
        self.private_t = None
        self.logits_s = None
        self.logit_t = None
        self.shared_feat = None
        self.shared_inverse_feat = None
        self.logits_d = None
        self.target_concat_feat = None
        self.source_concat_feat = None
        self.decode_s = None
        self.decode_t = None
    
    def concat_operation(self, shared_repr, private_repr):
        return shared_repr + private_repr

    def forward(self, source_images, target_images):
        # 获得他们公共的部分
        self.shared_s = self.shared_encoder(source_images)
        # self.shared_s = F.normalize(self.shared_s, p=2, dim=1)
        self.shared_t = self.shared_encoder(target_images)
        # self.shared_t = F.normalize(self.shared_t, p=2, dim=1)
        # 获得他们私有的各自部分
        self.private_s = self.private_source_encoder(source_images)
        # self.private_s = F.normalize(self.private_s, p=2, dim=1)
        self.private_t = self.private_target_encoder(target_images)
        # self.private_t = F.normalize(self.private_t, p=2, dim=1)
        self.logits_s = self.classfication_head(self.shared_s)
        self.logits_t = self.classfication_head(self.shared_t)

        # 将源域与目标域的共享特征进行拼接
        self.shared_feat = torch.cat([self.shared_s, self.shared_t], dim = 0)
        # 对拼接的特征首先来一个来一个梯度反转层
        if getattr(self, 'grl', None) is not None:
            self.shared_inverse_feat = self.grl(self.shared_feat)
        # 之后送进域判别器
        self.logits_d = self.domain_classfication(self.shared_inverse_feat)

        # 下面进入网络重建的部分
        # 改进之前的
        self.target_concat_feat=self.concat_operation(self.shared_t,self.private_t)
        self.source_concat_feat =self.concat_operation(self.shared_s,self.private_s)
        # 改进之后的
        # self.target_concat_feat=self.concat_operation(self.shared_s,self.private_t)
        # self.source_concat_feat =self.concat_operation(self.shared_t,self.private_s)

        self.decode_s = self.shared_decoder(self.source_concat_feat)
        self.decode_t = self.shared_decoder(self.target_concat_feat)

        return

# 这里编写一个损失函数
def calc_losses(dsn, source_images, target_images, source_labels, domain_labels):
    # 这里需要注意的是，传入的source_labels和domain_labels都是已经编码为one hot 向量了
    source_labels = torch.argmax(source_labels, dim=1)

    # print("source_labels:", source_labels)

    domain_labels = torch.argmax(domain_labels, dim=1)
    # print("domain_labels:", domain_labels)
    # 首次计算源域上的分类损失与域判别损失
    criterion1 = torch.nn.CrossEntropyLoss()
    # 分类损失
    source_class_loss = criterion1(dsn.logits_s, source_labels)
    # 域判别损失
    criterion2 = torch.nn.CrossEntropyLoss()
    similarity_loss = criterion2(dsn.logits_d, domain_labels)
    # 计算差异性损失
    # 目标差异性损失
    target_diff_loss =  difference_loss(dsn.shared_t, dsn.private_t, weight=0.05)
    # 源域差异性损失
    source_diff_loss = difference_loss(dsn.shared_s, dsn.private_s, weight=0.05)
    source_recon_loss = torch.nn.functional.mse_loss(source_images, dsn.decode_s)
    target_recon_loss = torch.nn.functional.mse_loss(target_images, dsn.decode_t)

    total_loss = source_class_loss + similarity_loss + target_diff_loss + source_diff_loss + source_recon_loss + target_recon_loss 

    return source_class_loss, similarity_loss, target_diff_loss, source_diff_loss,source_recon_loss, target_recon_loss, total_loss


def difference_loss(private_samples, shared_samples, weight=0.05):
    # print("private_samples", private_samples.shape)
    # print("shared_samples", shared_samples.shape)
    private_samples = private_samples - torch.mean(private_samples, dim=0, keepdim=True)
    shared_samples = private_samples - torch.mean(shared_samples, dim=0, keepdim=True)
    # print("private_samples", private_samples.shape)
    # print("shared_samples", shared_samples.shape)
    private_samples = torch.nn.functional.normalize(private_samples, p=2, dim=1)
    shared_samples = torch.nn.functional.normalize(shared_samples, p=2, dim=1)
    # print("private_samples", private_samples.shape)
    # print("shared_samples", shared_samples.shape)
    correlation_matrix = torch.matmul(private_samples.transpose(0, 1), shared_samples)
    # print("correlation_matrix", correlation_matrix.shape)
    cost = torch.mean(torch.square(correlation_matrix)) * weight
    # print("cost", cost.shape)
    cost = torch.where(cost > 0, cost, 0)
    # print("cost", cost.shape)
    return cost

def train(max_epoch, optimizer, scheduler, dsn, batch_size):

    train_dir = "./data"
    
    
    norm_mean = calc_mean(train_dir)
    norm_std = [1.0, 1.0, 1.0]

    train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建MyDataset实例
    train_data = Mnist_and_MnistM_Dataset(data_dir=train_dir, transform=train_transform, mode = 'train')
    

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size//2, shuffle=True)

    domain_labels = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),
                               np.tile([0., 1.], [batch_size // 2, 1])])
    domain_labels = torch.tensor(domain_labels)
    # print("domain_labels", domain_labels)
    domain_labels = domain_labels.to(device)
    # print("domain_labels", domain_labels)

    for epoch in range(max_epoch):
        # 因为网络中有BN层因此，最开始将网络的模式改为'train'模式
        # print("train模式设定之前")
        dsn.train()
        # print("train模式设定完毕")
        for i, data in enumerate(train_loader):
            # print("{}/{}".format(epoch, i))
            batch_source, batch_target, batch_y_s, batch_y_t = data

            # 这里需要将pytorch的tensor转换为图像，显示出来
            # show(batch_source, batch_target, batch_y_s, batch_y_t, train_transform)
            # 这里观察了一下数据是没有问题的

            batch_source, batch_target, batch_y_s, batch_y_t = batch_source.to(device), batch_target.to(device), batch_y_s.to(device), batch_y_t.to(device)
            # print("batch_source:", batch_source.shape)
            # print("batch_target:", batch_target.shape)
            # print("batch_y_s:", batch_y_s.shape)

            # 优化器清除梯度
            optimizer.zero_grad()

            # forward + backward + update
            dsn(batch_source, batch_target)
            # print(outputs)
            source_class_loss, similarity_loss, target_diff_loss, source_diff_loss,source_recon_loss, target_recon_loss, total_loss =\
            calc_losses(dsn, batch_source, batch_target, batch_y_s, domain_labels)
            total_loss.backward()
            optimizer.step()

            # if i % 50 == 0:
            #     print("[{}/{}] [{}/{}] source_class_loss = {:.5f}, similarity_loss ={:.5f}, target_diff_loss ={:.5f}, source_diff_loss={:.5f},source_recon_loss={:.5f},target_recon_loss ={:.5f}, total_loss={:.5f}".format(\
            #         epoch, max_epoch, i, len(train_data)//(batch_size//2), source_class_loss.detach().cpu().numpy(), \
            #     similarity_loss.detach().cpu().numpy(),
            #     target_diff_loss.detach().cpu().numpy(), source_diff_loss.detach().cpu().numpy(),\
            #     source_recon_loss.detach().cpu().numpy(), target_recon_loss.detach().cpu().numpy(), total_loss.detach().cpu().numpy()))

            # 替换原有的print语句
            if i % 50 == 0:
                log_message = (
                    f"[{epoch}/{max_epoch}] [{i}/{len(train_data)//(batch_size//2)}] "
                    f"source_class_loss = {source_class_loss.detach().cpu().numpy():.5f}, "
                    f"similarity_loss = {similarity_loss.detach().cpu().numpy():.5f}, "
                    f"target_diff_loss = {target_diff_loss.detach().cpu().numpy():.5f}, "
                    f"source_diff_loss = {source_diff_loss.detach().cpu().numpy():.5f}, "
                    f"source_recon_loss = {source_recon_loss.detach().cpu().numpy():.5f}, "
                    f"target_recon_loss = {target_recon_loss.detach().cpu().numpy():.5f}, "
                    f"total_loss = {total_loss.detach().cpu().numpy():.5f}"
                    )
                logging.info(log_message)
        # 更新学习率
        scheduler.step() 
        # 模型进行测试
        evaluation(500, dsn, epoch, max_epoch) 


def evaluation(test_batchsize, dsn, epoch, max_epoch):
    # 这里主要评估三部分指标，分别是源域上的分类器性能指标(测试集)、目标域上的分类器指标(测试集)、dann网络的分类准确率(测试集)
    test_dir = "./data"
    
    norm_mean = calc_mean(test_dir)
    norm_std = [1.0, 1.0, 1.0]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    domain_labels = np.vstack([np.tile([1., 0.], [test_batchsize // 2, 1]),
                               np.tile([0., 1.], [test_batchsize // 2, 1])])
    domain_labels = torch.tensor(domain_labels)
    domain_labels = domain_labels.to(device)

    test_data = Mnist_and_MnistM_Dataset(data_dir=test_dir, transform=test_transform, mode ='test')
    test_loader = DataLoader(dataset=test_data, batch_size=test_batchsize//2)
    
    total = 0
    correct_source_pred = 0
    correct_target_pred = 0
    correct_domain_pred = 0

    with torch.no_grad():
        # 模型进入到测试模式，主要影响模型的BN层的操作
        dsn.eval()
        for i, data in enumerate(test_loader):
            batch_source, batch_target, batch_y_s, batch_y_t = data
            batch_source, batch_target, batch_y_s, batch_y_t = batch_source.to(device), batch_target.to(device), batch_y_s.to(device), batch_y_t.to(device)
            # 改变一次total的数值
            total = total + batch_source.shape[0]
            # 模型先进行一次前向传播
            dsn(batch_source, batch_target)
            # source accuracy
            source_pred = torch.nn.Softmax(dim =1)(dsn.logits_s)
            
            correct_source_pred += (torch.argmax(source_pred, 1) == torch.argmax(batch_y_s, 1)).sum().item()
            
            # target accuracy
            target_pred = torch.nn.Softmax(dim =1)(dsn.logits_t)
            correct_target_pred += (torch.argmax(target_pred, 1) == torch.argmax(batch_y_t, 1)).sum().item()
            

            # domain accuracy
            domain_pred = torch.nn.Softmax(dim =1)(dsn.logits_d)
            correct_domain_pred += (torch.argmax(domain_pred, 1) == torch.argmax(domain_labels, 1)).sum().item()
            
    # print ("[{}/{}] 源域测试集分类准确率source_acc:{:.5f}, 目标域测试集分类准确率target_acc:{:.5f}, 测试集上域判别准确率domain_acc:{:.5f}".format(\
    #     epoch, max_epoch, correct_source_pred/total, correct_target_pred/total, correct_domain_pred/(total*2)))
    log_message = (
        f"[{epoch}/{max_epoch}] "
        f"Source domain test set classification accuracy:{correct_source_pred/total:.5f}, "
        f"Target domain test set classification accuracy:{correct_target_pred/total:.5f}, "
        f"Domain discrimination accuracy on test set:{correct_domain_pred/(total*2):.5f}"
    )
    logging.info(log_message)

if __name__ == '__main__': 

    # input = torch.rand([10, 3, 28, 28])
    # input = torch.rand([20, 100])
    # encode = Shared_Encode()
    # private_target_encode = Private_Target_Encoder()
    # private_source_encode = Private_Source_Encoder()
    # shared_decoder = Shared_Decoder(28, 28, 3)
    # output = encode(input)
    # output = private_target_encode(input)
    # output = shared_decoder(input)

    # 设定整体的batch_size 为20
    batch_size = 550
    # 设定两个输入图像
    # source_images = torch.rand([batch_size, 3, 28, 28])
    # target_images = torch.rand([batch_size, 3, 28, 28])
    # source_labels = torch.rand([batch_size, 10])
    # domain_labels = torch.rand([2 * batch_size, 2])

    # 首次设定种子,设定种子必须要在搭建网络之前,设定种子需要在实例化网络之前
    seed = 1234
    set_seed(seed)

    # 构造子网络
    shared_encoder = Shared_Encode()
    private_source_encoder = Private_Source_Encoder()
    private_target_encoder = Private_Target_Encoder()
    shared_decoder = Shared_Decoder(28, 28, 3)
    classfication_head = Classfication_Head()
    domain_classfication = Domain_Classfication()
    dsn = DSN(shared_encoder, private_source_encoder, private_target_encoder, shared_decoder, classfication_head, domain_classfication, use_GRL=1)

    # 将网络传到gpu上
    dsn = dsn.to(device)
    # print("dsn", dsn)
    # 先进行一次前向传播
    # dsn(source_images, target_images)
    # 在这里完成losses的计算
    # source_class_loss, similarity_loss, target_diff_loss, source_diff_loss,source_recon_loss, target_recon_loss, total_loss =\
    #     calc_losses(dsn, source_images, target_images, source_labels, domain_labels)
    # print("source_labels:", source_labels.shape)
    # source_labels = torch.argmax(source_labels, dim=1)
    # domain_labels = torch.argmax(domain_labels, dim=1)

    # print("source_labels:", source_labels.shape)
    # print("source_labels:", source_labels)

    # print("source_class_loss = {}, similarity_loss ={}, target_diff_loss ={}, source_diff_loss={},source_recon_loss={},\
    #  target_recon_loss ={}, total_loss={}".format(source_class_loss, similarity_loss, target_diff_loss, source_diff_loss,\
    #      source_recon_loss, target_recon_loss, total_loss))
    
    # 定义优化器
    optimizer = torch.optim.Adam(dsn.parameters(), lr=0.005, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8) 
    train(60, optimizer, scheduler, dsn, batch_size)


    

    

    


