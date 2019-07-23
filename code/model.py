import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from utils import Get_cifar10_Data
from torchvision import models

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(nn.Linear(512,128),nn.Dropout(0.5),nn.ReLU(),
                                nn.Linear(128,2))
        

    
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # FC
        x = self.avg(e4)
        x = x.view(x.size(0), -1)#avg后必须得有一步。
        
        output = self.fc(x)

        
        return F.sigmoid(output)
if __name__ == '__main__':
    input = torch.randn(2,3,224,224)
    model =ResNet34()
    out = model(input)
    print(out.shape,out)
