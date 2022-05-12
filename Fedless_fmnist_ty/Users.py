import binascii
import json
import random
import sys
import time
from io import BytesIO
import numpy as np

import torch

from AccTest import test_Acc
from LocalTrain import LocalTrain
from Initial import para, DataSize, Logger, init_cnn, para_quit, random_int_list, quit_random, addtwodimdict, NpEncoder
from Initial import Net

sys.stdout = Logger('log_user.txt')

user_number, total_round = para()
user_number_base = user_number
quitMode, quit_range = para_quit()

# Each client creates a random seed corresponding to the server
GlobalModel = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global user_number, time0, time1, flag
    t_acc = np.zeros((total_round,))
    quitNum = 0
    quitId = []
    quitNum_e = 0
    quitId_e = []
    quit_seed = 6
    secert_seed = 6

    # secrets = get_secrets(user_number)  # 得到mask， 并将mask给ss
    # masks_str = {}
    # # shares = np.zeros((user_number, user_number), dtype=str)
    # shares = dict()  # user j持有的user i的份额

    if quitMode == 1:
        quitNum = user_number * quit_range // 100
        quitId = random_int_list(0, user_number - 1, quitNum, 1)
        # print(quitId)

    time_avg_computation = 0
    size_avg_communication = 0

    for e in range(total_round):
        print('*****************')
        print('Start round %d' % int(e + 1))
        print('*****************')

        quitId_new = []

        # quit users
        if quitMode == 1 and quitNum_e < quitNum:
            quit_user_num = quit_random(((quitNum - quitNum_e) // 5) + 1, quit_seed)
            for i in range(quitNum_e, quitNum_e + quit_user_num):
                # quitId_e.append(quitId[i])
                quitId_new.append(quitId[i])
            quitNum_e += quit_user_num
            quit_seed += 1
        print('The number of users in round %d: %d' % (int(e + 1), user_number - quitNum_e))

        acc = np.zeros((user_number,))
        a_acc = 0.0
        decode = init_cnn()

        size_communication = 0
        time_computation = 0
        time_cryptographic = 0
        flag = 1

        for j in range(user_number):
            # Local Train
            if j == 0:
                print("users start training")
            if j in quitId_new or j in quitId_e:
                continue

            data, acc[j] = LocalTrain(GlobalModel, j)
            print('user%d--acc--%f' % (j + 1, acc[j]))
            a_acc += acc[j]

            if flag == 1:
                for d in range(8):
                    send_data = json.dumps(data[d], cls=NpEncoder)
                    size_communication += (len(send_data) / float(1024*1024))
                flag = 0

            time0 = time.perf_counter()
            decode[0] += data[0] * (1 / (user_number - quitNum_e))
            decode[1] += data[1] * (1 / (user_number - quitNum_e))
            decode[2] += data[2] * (1 / (user_number - quitNum_e))
            decode[3] += data[3] * (1 / (user_number - quitNum_e))
            decode[4] += data[4] * (1 / (user_number - quitNum_e))
            decode[5] += data[5] * (1 / (user_number - quitNum_e))
            decode[6] += data[6] * (1 / (user_number - quitNum_e))
            decode[7] += data[7] * (1 / (user_number - quitNum_e))
            time1 = time.perf_counter()
            time_computation += (time1 - time0)
        print('Global model generated')

        for i in range(len(quitId_new)):
            quitId_e.append(quitId_new[i])

        time_computation = time_computation * 1000
        time_avg_computation += time_computation
        size_avg_communication += size_communication
        print('Computational cost per user: %f ms' % time_computation)
        print('Communication cost per user: %f MB' % size_communication)

        for i in GlobalModel.state_dict():
            if device == 'cpu':
                GlobalModel.state_dict()[i] -= GlobalModel.to(device).state_dict()[i]
            else:
                GlobalModel.state_dict()[i] -= GlobalModel.state_dict()[i]
        torch.save(GlobalModel.state_dict(),
                   "./model_state_dict_" + ".pt")
        model_dict = torch.load("model_state_dict_" + ".pt")
        model_dict['fc1.weight'] += torch.from_numpy(decode[0])
        model_dict['fc1.bias'] += torch.from_numpy(decode[1])
        model_dict['fc2.weight'] += torch.from_numpy(decode[2])
        model_dict['fc2.bias'] += torch.from_numpy(decode[3])
        model_dict['fc3.weight'] += torch.from_numpy(decode[4])
        model_dict['fc3.bias'] += torch.from_numpy(decode[5])
        model_dict['fc4.weight'] += torch.from_numpy(decode[6])
        model_dict['fc4.bias'] += torch.from_numpy(decode[7])

        GlobalModel.load_state_dict(model_dict)
        g_acc = test_Acc(GlobalModel)
        print('Global accuracy---%f' % g_acc)
        t_acc[e] += (g_acc)

    print('+++++++++++')
    print('Average computational cost of a user per round: %f ms' % (time_avg_computation / total_round))
    print('Average communication cost of a user per round: %f MB' % (size_avg_communication / total_round))
    print('+++++++++++')
    print('Accs:')
    print(t_acc)
    print('+++++++++++')


if __name__ == '__main__':
    main()
