import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.models import resnet34
from tensorboardX import SummaryWriter

from utils import My_Dataset
from utils import Get_cifar10_Data
from model import *

#----------------------------------property of training------------------------
device = torch.device('cuda')
dtype = torch.float32
vis_dir = './vis'
writer = SummaryWriter(vis_dir)

batch_size = 32
#the batch size of train and validate is same,
#which can be modified in function of Get_cifar10_Data
epoch = 50
per_iter = 100
#------------------------------------get data----------------------------------
train_loader,val_loader = Get_cifar10_Data(batch_size)
    #the transform of data is defined in the function,
    #whcih has resize(32),crop(32),normalize,and ToTesnor
#------------------------------------get model---------------------------------
model = ResNet34()
model = model.to(device=device)
#------------------------------------get other machine of train----------------

cretirien = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
schedual = torch.optim.lr_scheduler.StepLR(optimizer,20)
#---------------------------------training-------------------------------------
def check_accuracy(loader, model):  
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Validata_data Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        
    return acc
loss_his = []
train_acc_his = []
val_acc_his = []
for e in range(epoch):
    schedual.step()
    loss_sigma = 0.0
    correct_num = 0
    samples_num = 0
    
    for t,(x,y) in enumerate(train_loader):
        #此处调用dataloader有两个方法，一个是用enumerate，可以直接产生t
        #另一个是直接for in train_loader这样也会循环batch的数据，但是没有t了
        #还可以直接求train_loader的长度，那是不是也可以索引嗯
        model.train()
        x=x.to(device=device,dtype = dtype)
        y=y.to(device=device,dtype = torch.long)
        
        outputs = model(x)
        _, pred = outputs.max(1)
        correct_num += (pred==y).sum()

        samples_num += pred.size(0)
        
        print(outputs.shape)
        loss = cretirien(outputs,y)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        loss_sigma += loss.item()
        optimizer.step()
        #此处多了一个loss_his，每100iter记录一次loss，loss_sigma几率所有的loss加和，100次清一次，目的为求loss_avg
        if t % per_iter == 0:
            loss_avg = loss_sigma / 10
            loss_his.append(loss_avg)
            loss_sigma = 0.0
            train_acc = float(correct_num) / samples_num
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{}".format(
                e + 1, epoch, t + 1, len(train_loader), loss_avg, train_acc))
            val_acc = check_accuracy(val_loader,model)
            val_acc_his.append(val_acc)
            train_acc_his.append(train_acc)
            writer.add_scalar('train_acc', train_acc, per_iter)
            writer.add_scalar('val_acc', val_acc, per_iter)
            writer.add_scalar('loss', loss_avg, per_iter)
            
            print()

#this is a code using finalshell

        
        
        
        
        
        
