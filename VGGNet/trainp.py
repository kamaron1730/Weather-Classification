import time
from torchvision import  transforms
from VGGNet.model import *
# import sys
# from tqdm import tqdm
import matplotlib.pyplot as plt
# from loaddata import *
from torch.utils.data import DataLoader

import torchvision

def main():
    n_epochs = 1
    #图片大小
    resize=224
    # 初始化验证机的最小误差为正无穷
    best_acc = 0.0

    net = VggNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print(device)
    # 定义损失函数和梯度下降优化器
    # 损失函数为交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器（随机梯度下降SGD优化器，学习率为0.001）
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # # 加载训练集、验证集
    # trainloader = LoadData('data_set\weather_classification', 32, 'train')
    # valloader = LoadData('data_set\weather_classification', 32, 'val')
    # train_set = DataLoader(trainloader, batch_size=32, shuffle=True, num_workers=0)
    # val_set = DataLoader(valloader, batch_size=32, shuffle=True, num_workers=0)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(resize),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(resize*1.25)),
                                   transforms.CenterCrop(resize),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    trainloader = torchvision.datasets.ImageFolder(root="data_set/train",
                                         transform=data_transform["train"])
    valloader = torchvision.datasets.ImageFolder(root="data_set/val",
                                                     transform=data_transform["val"])
    train_set = DataLoader(trainloader, batch_size=10, shuffle=True, num_workers=2)
    val_set = DataLoader(valloader, batch_size=10, shuffle=True, num_workers=2)

    # net.load_state_dict(torch.load('modelcnn3.pt'))
    train_loss_hist=[]
    val_loss_hist=[]
    val_acc_hist=[]
    for epoch in range(n_epochs):
        # 训练
        net.train()
        running_loss=0.0
        t1 = time.perf_counter()
        # train_bar =tqdm(train_set,file=sys.stdout)
        for i,data in enumerate(train_set):
            images,labels=data
            outputs=net(images.to(device))
            loss=criterion(outputs,labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            #print train process
            rate = (i+1)/len(train_set)
            a = "*" * int(rate * 50)
            b= "." * int((1-rate)*50)
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,n_epochs,loss)
            print("\rtrain loss : {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100),a,b,loss),end="")
        print()
        print(time.perf_counter()-t1)
        #验证
        net.eval()
        acc = 0.0
        valrunning_loss = 0.0
        with torch.no_grad():
            for val_data in val_set:
                val_images,val_labels=val_data
                val_outputs=net(val_images.to(device))
                val_loss=criterion(val_outputs,val_labels.to(device))
                valrunning_loss += val_loss.item()
                _,predicted=torch.max(val_outputs, 1)
                acc += torch.eq(predicted,val_labels.to(device)).sum().item()
        val_accurate = acc / len(val_set.dataset)
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f val_accuracy: %.3f' %(epoch + 1, running_loss /i, valrunning_loss/len(val_set),val_accurate))
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     torch.save(net.state_dict(), 'model/modelVgg1.pt')
        train_loss_hist.append(running_loss/len(train_set))
        val_loss_hist.append(valrunning_loss/len(val_set))
        val_acc_hist.append((val_accurate))

    print('Finished Training')

        #图像显示
        # plt.figure()
        # x,=plt.plot(train_loss_hist)
        # y,=plt.plot(val_loss_hist)
        # plt.legend([x,y],['train loss','val loss'])
        # plt.title('Train/Val loss')
        # plt.xlabel('#mini batch * 250')
        # plt.ylabel('Loss')
    plt.plot(train_loss_hist,label='train loss')
    plt.plot(val_loss_hist,label='val loss')
    plt.plot(val_acc_hist,label='val acc')
    plt.title("VggNet learning rate=0.01")
    plt.legend()
    plt.savefig('image/Vgg2.jpg')
    plt.show()
if __name__ == '__main__':
    main()