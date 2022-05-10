import torch
import torch.nn as nn
import torch.nn.functional as F


class VggNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(VggNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1), #input_size:224,224,3 output_size:224,224,64
            nn.Conv2d(64,64,kernel_size=3,padding=1),#224,224,64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2,kernel_size=2)#112,112,64
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),#112,112,128
            nn.Conv2d(128, 128, kernel_size=3, padding=1),#112,112,128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2)#56,56,128
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),#56,56,256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),#56,56,256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),#56,56,256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2)#28,28,256
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),#28,28,512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),#28,28,512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),#28,28,512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2)#14,14,512
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),#14,14,512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),#14,14,512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),#14,14,512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2)#7,7,512
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=torch.flatten(x,start_dim=1)
        x=self.classifier(x)
        x=F.log_softmax(x,dim=1)
        return x



