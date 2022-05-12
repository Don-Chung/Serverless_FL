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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def init_cnn():
    d = {}
    fc1_weight = np.zeros((1000, 784))
    d.setdefault(0, fc1_weight)
    fc1_bias = np.zeros((1000,))
    d.setdefault(1, fc1_bias)
    fc2_weight = np.zeros((500, 1000))
    d.setdefault(2, fc2_weight)
    fc2_bias = np.zeros((500,))
    d.setdefault(3, fc2_bias)
    fc3_weight = np.zeros((200, 500))
    d.setdefault(4, fc3_weight)
    fc3_bias = np.zeros((200,))
    d.setdefault(5, fc3_bias)
    fc4_weight = np.zeros((10, 200))
    d.setdefault(6, fc4_weight)
    fc4_bias = np.zeros((10,))
    d.setdefault(7, fc4_bias)
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
