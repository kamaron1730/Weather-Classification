import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from PIL import Image
import os
import json


def plot_class_preds(net,
                     images_dir: str,
                     transform,
                     num_plot: int = 5,
                     device="cpu"):
    if not os.path.exists(images_dir):
        print("not found {} path, ignore add figure.".format(images_dir))
        return None

    label_path = os.path.join(images_dir, "label.txt")
    if not os.path.exists(label_path):
        print("not found {} file, ignore add figure".format(label_path))
        return None

    # read class_indict
    json_label_path = '../class_indices.json'
    assert os.path.exists(json_label_path), "not found {}".format(json_label_path)
    json_file = open(json_label_path, 'r')
    # {"0": "daisy"}
    weather_class = json.load(json_file)
    # {"daisy": "0"}
    class_indices = dict((v, k) for k, v in weather_class.items())

    # reading label.txt file
    label_info = []
    with open(label_path, "r") as rd:
        for line in rd.readlines():
            line = line.strip()
            if len(line) > 0:
                split_info = [i for i in line.split(" ") if len(i) > 0]
                assert len(split_info) == 2, "label format error, expect file_name and class_name"
                image_name, class_name = split_info
                image_path = os.path.join(images_dir, image_name)
                # 如果文件不存在，则跳过
                if not os.path.exists(image_path):
                    print("not found {}, skip.".format(image_path))
                    continue
                # 如果读取的类别不在给定的类别内，则跳过
                if class_name not in class_indices.keys():
                    print("unrecognized category {}, skip".format(class_name))
                    continue
                label_info.append([image_path, class_name])

    if len(label_info) == 0:
        return None

    # get first num_plot info
    if len(label_info) > num_plot:
        label_info = label_info[:num_plot]

    num_imgs = len(label_info)
    images = []
    labels = []
    for img_path, class_name in label_info:
        # read img
        img = Image.open(img_path).convert("RGB")
        label_index = int(class_indices[class_name])

        # preprocessing
        img = transform(img)
        images.append(img)
        labels.append(label_index)

    # batching images
    images = torch.stack(images, dim=0).to(device)

    # inference
    with torch.no_grad():
        output = net(images)
        probs, preds = torch.max(torch.softmax(output, dim=1), dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()

    # width, height
    fig = plt.figure(figsize=(num_imgs * 2.5, 3), dpi=100)
    for i in range(num_imgs):
        # 1：子图共1行，num_imgs:子图共num_imgs列，当前绘制第i+1个子图
        ax = fig.add_subplot(1, num_imgs, i+1, xticks=[], yticks=[])

        # CHW -> HWC
        npimg = images[i].cpu().numpy().transpose(1, 2, 0)

        # 将图像还原至标准化之前
        # mean:[0.485, 0.456, 0.406], std:[0.229, 0.224, 0.225]
        npimg = (npimg * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        plt.imshow(npimg.astype('uint8'))

        title = "{}, {:.2f}%\n(label: {})".format(
           weather_class[str(preds[i])],  # predict class
            probs[i] * 100,  # predict probability
            weather_class[str(labels[i])]  # true class
        )
        ax.set_title(title, color=("green" if preds[i] == labels[i] else "red"))

    return fig
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    #data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 打印平均loss
        #data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
    #data_loader = tqdm(data_loader, desc="validation...", file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 计算预测正确的比例
    acc = sum_num.item() / num_samples

    return acc