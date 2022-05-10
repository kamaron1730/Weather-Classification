# Weather-Classification
# 基于深度神经网络的天气识别算法研究

***
### 在日常生活中，人类的各方面活动都与天气现象密切相关，天气识别在各种领域都有着巨大的应用潜力和价值。近年来，随着深度学习技术的飞速发展以及现代社会信息化程度的日益提高，人们对实现室外天气现象的自动检测和识别也越来也迫切。
### 本文基于深度神经网络对天气识别算法开展了研究，采用一个包括晴、多云、雨、雪、霾、雷电六类天气现象数据集(Multi-class Weather Dataset)，通过pytorch实现五种经典CNN模型，与迁移学习相结合，训练并挖掘天气数据的特征和内在规律，使用深度学习的方法实现天气的自动分类，具有很强的实用价值。
### 如何正确识别天气现象是一个不容忽视的问题，它不仅关系着人们的日常生活，而且还影响许多机器视觉系统是否能够正常运行。在很多应用场景下，实时获取当前环境的天气情况极为重要，所以在众多领域内使用机器视觉对天气识别都具备很大的研究价值和应用前景。

***

# 此文件里面是相关实验的代码
## 以下是文件的介绍：
1. 本实验搭建了五个卷积神经网络模型，分别是AlexNet、VGGNet、GoogLeNet、ResNet、MobileNet。
2. 每个模型的目录下有：

                     > model.py--------搭建模型结构

                     > train.py--------训练100轮，并且预测模型的训练效果，在tensorboard上可视化

                     > utils.py--------封装了训练、验证、预测方法
3. class_indices.json里面包含六种实验天气类别
4. data_random:在网络上随机寻找的天气图片，不在数据集中，用于预测模型训练效果
5. 数据集来源：[http://vcc.szu.edu.cn/file/upload_file/0/58/weboem_informations/classification.zip](http://vcc.szu.edu.cn/file/upload_file/0/58/weboem_informations/classification.zip)
6. 使用split_data.py将数据划分成训练集和验证集
