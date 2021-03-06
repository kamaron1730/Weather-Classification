import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2),  #imput_size:[224,224,3] output_size:[55,55,48]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),               #27,27,48
            nn.Conv2d(48, 128, kernel_size=5, padding=2),       #27,27,128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),              #13,13,128
            nn.Conv2d(128, 192, kernel_size=3, padding=1),      #13,13,192
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),      #13,13,192
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),      #13,13,128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),              #6,6,128
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
    def forward(self,x):
        x = self.features(x)
        x=torch.flatten(x,start_dim=1)  #展平处理从索引1开始 (channel)
        x = self.classifier(x)
        return x
