import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader


def test_Acc(GlobalModel):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_net = copy.deepcopy(GlobalModel).to(device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              transform=transform)
    batch_size = 32
    testloader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    # 构造测试的dataloader
    dataiter = iter(testloader)
    # 预测正确的数量和总数量
    correct = 0
    total = 0
    # 使用torch.no_grad的话在前向传播中不记录梯度，节省内存
    # cv2.namedWindow('predictPic', cv2.WINDOW_NORMAL)
    to_pil_image = transforms.ToPILImage()
    with torch.no_grad():
        for images, labels in dataiter:
            images, labels = images.to(device), labels.to(device)
            # 预测
            outputs = model_net(images)
            # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
