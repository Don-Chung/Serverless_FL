import pickle
import os
import sys

import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random

from Initial import para, para_quit, get_FileSize, Logger

sys.stdout = Logger('log.txt')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

#data_of_one_batch = 100  # 每批数据量


def create_users(num, dir, seed):
    '''
    This function creates clients that hold non-iid MNIST data
    but the way these indices are grouped, they create a non-iid client.)
    '''
    global z
    if not os.path.exists(dir):
        os.makedirs(dir)

    multi = 5
    size_o = (50000 // num) * multi

    dataset1 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    index = 0
    for i in range(num):
        if os.path.exists(dir + '/' + str(i + 1) + '_user.pkl'):
            print('Client exists at: ' + dir + '/' + str(i) + '_user.pkl')
            size = get_FileSize('./data/users/' + str(i + 1) + '_user.pkl')
            print('user -- %d --size: %f MB' % (i + 1, size))
            continue

        random.seed(seed)
        size = random.randint(size_o // 2, size_o)
        train_loader = DataLoader(dataset1, batch_size=size, shuffle=True)
        # index = random.randint(0, 50000 // size)
        for batch_idx, z in enumerate(train_loader):
            if batch_idx == index:
                break
        index += 1
        filehandler = open(dir + '/' + str(i + 1) + '_user.pkl', "wb")
        pickle.dump(z, filehandler)
        filehandler.close()
        print('client created at: ' + dir + '/' + str(i + 1) + '_user.pkl')
        size = get_FileSize('./data/users/' + str(i + 1) + '_user.pkl')
        print('user -- %d --size: %f MB' % (i + 1, size))
        seed += 1


if __name__ == '__main__':
    user_number, total_round = para()
    # quitMode, quit_range = para_quit()
    # user_number = 100
    # if quitMode == 1:
    #     num = user_number + (user_number * quit_range // 100)
    # else:
    #     num = user_number
    # for j in range(1, num + 1):
    create_users(user_number, os.getcwd() + '/data/users', 1)
        # size = get_FileSize('./data/users/' + str(j) + '_user.pkl')
        # print('user -- %d --size: %f MB' % (j, size))
