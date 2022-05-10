import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
# import torchvision.models.resnet
from VGGNet.model import VggNet
from torch.utils.data import DataLoader
import math
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import os
from VGGNet.utils import train_one_epoch, evaluate, plot_class_preds
import argparse

def main(args):
    n_epochs = 10
    # 图片大小
    resize = 224
    # 初始化验证机的最小误差为正无穷
    best_acc = 0.0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir="runs/weather_experiment")
    print("111")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(resize),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(resize * 1.25)),
                                   transforms.CenterCrop(resize),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    trainloader = torchvision.datasets.ImageFolder(root="../data_set/train",
                                                   transform=data_transform["train"])
    valloader = torchvision.datasets.ImageFolder(root="../data_set/val",
                                                 transform=data_transform["val"])
    batch_size = args.batch_size
    # 计算使用num_workers的数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_set = DataLoader(trainloader, batch_size=batch_size, shuffle=True, num_workers=0)
    val_set = DataLoader(valloader, batch_size=batch_size, shuffle=False, num_workers=0)
    net = VggNet(num_classes=args.num_classes).to(device)
    # 将模型写入tensorboard
    init_img = torch.zeros((1, 3, 224, 224), device=device)
    tb_writer.add_graph(net, init_img)
    # 如果存在预训练权重则载入
    if os.path.exists(args.weights):
        weights_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if net.state_dict()[k].numel() == v.numel()}
        net.load_state_dict(load_weights_dict, strict=False)
    else:
        print("not using pretrain-weights.")

    # 是否冻结权重
    if args.freeze_layers:
        print("freeze layers except fc layer.")
        for name, para in net.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    # model_weight_path = "../model/resnet34-pre.pth"
    # net.load_state_dict(torch.load(model_weight_path,map_location=device))
    #
    # inchannel = net.fc.in_features
    # net.fc = nn.Linear(inchannel,6)
    # 定义损失函数和梯度下降优化器
    # 损失函数为交叉熵损失函数

    criterion = nn.CrossEntropyLoss()

    # 定义优化器（随机梯度下降SGD优化器，学习率为0.001）
    pg = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=net,
                                    optimizer=optimizer,
                                    data_loader=train_set,
                                    device=device,
                                    epoch=epoch)
        # update learning rate
        scheduler.step()

        # validate
        acc = evaluate(model=net,
                       data_loader=val_set,
                       device=device)

        # add loss, acc and lr into tensorboard
        print("[epoch {}] accuracy: {}".format(epoch+1, round(acc, 3)))
        tags = ["train_loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch+1)
        tb_writer.add_scalar(tags[1], acc, epoch+1)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch+1)
        # add figure into tensorboard
        fig = plot_class_preds(net=net,
                               images_dir="../data_random",
                               transform=data_transform["val"],
                               num_plot=5,
                               device=device)
        if fig is not None:
            tb_writer.add_figure("predictions vs. actuals",
                                 figure=fig,
                                 global_step=epoch+1)

        # add conv1 weights into tensorboard
        tb_writer.add_histogram(tag="conv1",
                                values=net.conv1.weight,
                                global_step=epoch+1)
        tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=net.layer1[0].conv1.weight,
                                global_step=epoch+1)
        # save weights
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), "./weights/modelVGGNet.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    img_root = "../data_set/weather_classification"
    parser.add_argument('--data-path', type=str, default=img_root)

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='vggnet.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)