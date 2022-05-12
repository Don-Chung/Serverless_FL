import configparser
import json
import os
import random
import socket
import sys

import torch.nn as nn  # 指定torch.nn别名nn
import torch.nn.functional as F  # 引用神经网络常用函数包，不具有可学习的参数
import numpy as np

# from KeyAgreement import calculation

parent_dir = os.path.dirname(os.path.abspath(__file__))

config = configparser.ConfigParser()
config.read(parent_dir + "/para.ini")

# Parameter initialization
user_number = int(config.get("paras", "user_number"))  # Number of users
total_round = int(config.get("paras", "total_round"))  # Number of running rounds
quitMode = int(config.get("paras", "quitMode"))
quit_range = int(config.get("paras", "quit_range"))

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Net(nn.Module):
    def __init__(self, vgg_name):
        super(Net, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11():
    return Net('VGG11')


def VGG13():
    return Net('VGG13')


def VGG16():
    return Net('VGG16')


def VGG19():
    return Net('VGG19')


def init_cnn():
    d = {}
    # 0
    # features_0_weight = np.zeros((192, 9))
    features_0_weight = np.zeros((64, 3, 3, 3))
    d.setdefault(0, features_0_weight)
    features_0_bias = np.zeros((64,))
    d.setdefault(1, features_0_bias)
    features_1_weight = np.zeros((64,))
    d.setdefault(2, features_1_weight)
    features_1_bias = np.zeros((64,))
    d.setdefault(3, features_1_bias)
    # 4
    # features_3_weight = np.zeros((, 9))
    features_3_weight = np.zeros((64, 64, 3, 3))
    d.setdefault(4, features_3_weight)
    features_3_bias = np.zeros((64,))
    d.setdefault(5, features_3_bias)
    features_4_weight = np.zeros((64,))
    d.setdefault(6, features_4_weight)
    features_4_bias = np.zeros((64,))
    d.setdefault(7, features_4_bias)
    # 8
    # features_7_weight = np.zeros((, 9))
    features_7_weight = np.zeros((128, 64, 3, 3))
    d.setdefault(8, features_7_weight)
    features_7_bias = np.zeros((128,))
    d.setdefault(9, features_7_bias)
    features_8_weight = np.zeros((128,))
    d.setdefault(10, features_8_weight)
    features_8_bias = np.zeros((128,))
    d.setdefault(11, features_8_bias)
    # 12
    # features_10_weight = np.zeros((16384, 9))
    features_10_weight = np.zeros((128, 128, 3, 3))
    d.setdefault(12, features_10_weight)
    features_10_bias = np.zeros((128,))
    d.setdefault(13, features_10_bias)
    features_11_weight = np.zeros((128,))
    d.setdefault(14, features_11_weight)
    features_11_bias = np.zeros((128,))
    d.setdefault(15, features_11_bias)
    # 16
    # features_14_weight = np.zeros((32768, 9))
    features_14_weight = np.zeros((256, 128, 3, 3))
    d.setdefault(16, features_14_weight)
    features_14_bias = np.zeros((256,))
    d.setdefault(17, features_14_bias)
    features_15_weight = np.zeros((256,))
    d.setdefault(18, features_15_weight)
    features_15_bias = np.zeros((256,))
    d.setdefault(19, features_15_bias)
    # 20
    # features_17_weight = np.zeros((65536, 9))
    features_17_weight = np.zeros((256, 256, 3, 3))
    d.setdefault(20, features_17_weight)
    features_17_bias = np.zeros((256,))
    d.setdefault(21, features_17_bias)
    features_18_weight = np.zeros((256,))
    d.setdefault(22, features_18_weight)
    features_18_bias = np.zeros((256,))
    d.setdefault(23, features_18_bias)
    # 24
    # features_20_weight = np.zeros((65536, 9))
    features_20_weight = np.zeros((256, 256, 3, 3))
    d.setdefault(24, features_20_weight)
    features_20_bias = np.zeros((256,))
    d.setdefault(25, features_20_bias)
    features_21_weight = np.zeros((256,))
    d.setdefault(26, features_21_weight)
    features_21_bias = np.zeros((256,))
    d.setdefault(27, features_21_bias)
    # 28
    # features_24_weight = np.zeros((131072, 9))
    features_24_weight = np.zeros((512, 256, 3, 3))
    d.setdefault(28, features_24_weight)
    features_24_bias = np.zeros((512,))
    d.setdefault(29, features_24_bias)
    features_25_weight = np.zeros((512,))
    d.setdefault(30, features_25_weight)
    features_25_bias = np.zeros((512,))
    d.setdefault(31, features_25_bias)
    # 32
    # features_27_weight = np.zeros((262144, 9))
    features_27_weight = np.zeros((512, 512, 3, 3))
    d.setdefault(32, features_27_weight)
    features_27_bias = np.zeros((512,))
    d.setdefault(33, features_27_bias)
    features_28_weight = np.zeros((512,))
    d.setdefault(34, features_28_weight)
    features_28_bias = np.zeros((512,))
    d.setdefault(35, features_28_bias)
    # 36
    # features_30_weight = np.zeros((262144, 9))
    features_30_weight = np.zeros((512, 512, 3, 3))
    d.setdefault(36, features_30_weight)
    features_30_bias = np.zeros((512,))
    d.setdefault(37, features_30_bias)
    features_31_weight = np.zeros((512,))
    d.setdefault(38, features_31_weight)
    features_31_bias = np.zeros((512,))
    d.setdefault(39, features_31_bias)
    # 40
    # features_34_weight = np.zeros((262144, 9))
    features_34_weight = np.zeros((512, 512, 3, 3))
    d.setdefault(40, features_34_weight)
    features_34_bias = np.zeros((512,))
    d.setdefault(41, features_34_bias)
    features_35_weight = np.zeros((512,))
    d.setdefault(42, features_35_weight)
    features_35_bias = np.zeros((512,))
    d.setdefault(43, features_35_bias)
    # 44
    # features_37_weight = np.zeros((262144, 9))
    features_37_weight = np.zeros((512, 512, 3, 3))
    d.setdefault(44, features_37_weight)
    features_37_bias = np.zeros((512,))
    d.setdefault(45, features_37_bias)
    features_38_weight = np.zeros((512,))
    d.setdefault(46, features_38_weight)
    features_38_bias = np.zeros((512,))
    d.setdefault(47, features_38_bias)
    # 48
    # features_40_weight = np.zeros((262144, 9))
    features_40_weight = np.zeros((512, 512, 3, 3))
    d.setdefault(48, features_40_weight)
    features_40_bias = np.zeros((512,))
    d.setdefault(49, features_40_bias)
    features_41_weight = np.zeros((512,))
    d.setdefault(50, features_41_weight)
    features_41_bias = np.zeros((512,))
    d.setdefault(51, features_41_bias)

    classifier_weight = np.zeros((10, 512))
    d.setdefault(52, classifier_weight)
    classifier_bias = np.zeros((10,))
    d.setdefault(53, classifier_bias)
    return d


def para():
    return user_number, total_round


def para_quit():
    return quitMode, quit_range


def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024 * 1024)

    return round(fsize, 2)


# 解析JSON
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


SizeSet = {}


# 每个客户的数据集大小
class DataSize:
    def size_of_data(worker_num):
        SizeSet = []
        for i in range(worker_num):
            tmp = get_FileSize('./data/users/' + str(i + 1) + '_user.pkl')
            SizeSet.append(tmp)
        return SizeSet


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def addtwodimdict(thedict, key_a, key_b, val):
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a: {key_b: val}})


def random_int_list(start, stop, length, seed):
    random.seed(seed)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        tmp = random.randint(start, stop)
        while tmp in random_list:
            tmp = random.randint(start, stop)
        random_list.append(tmp)
    return random_list


def quit_random(quit_num, seed):
    random.seed(seed)
    flag = random.randint(0, 1)
    if flag == 1:
        return random.randint(1, quit_num)
    else:
        return 0
