import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from GoogLeNet.utils import plot_class_preds
from GoogLeNet.model import GoogLeNet
from torch.utils.tensorboard import SummaryWriter


def main():
    epochs = 100
    resize=224
    writer = SummaryWriter(log_dir="runs/weather_experiment")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(resize),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(resize * 1.25)),
                                   transforms.CenterCrop(resize),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = datasets.ImageFolder(root="../data_set/train",
                                                   transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root="../data_set/val",
                                                 transform=data_transform["val"])
    train_num = len(train_dataset)

    weather_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in weather_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

 
    net = GoogLeNet(num_classes=6, aux_logits=True, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)


    best_acc = 0.0
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    save_path = './weights/googleNet.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        #train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            logits, aux_logits2, aux_logits1 = net(images.to(device))
            loss0 = loss_function(logits, labels.to(device))
            loss1 = loss_function(aux_logits1, labels.to(device))
            loss2 = loss_function(aux_logits2, labels.to(device))
            loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()


        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            #val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        writer.add_scalar("train_loss",running_loss / train_steps)
        writer.add_scalar("accuracy",val_accurate)
        writer.flush()
        fig = plot_class_preds(net=net,
                               images_dir="data_random",
                               transform=data_transform["val"],
                               num_plot=8,
                               device=device)
        if fig is not None:
            writer.add_figure("predictions vs. actuals",
                                 figure=fig,
                                 global_step=epoch)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
