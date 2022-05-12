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
from Initial import para, DataSize, Logger, init_cnn, para_quit, random_int_list, quit_random, addtwodimdict, NpEncoder, \
    VGG16

sys.stdout = Logger('log_user.txt')

user_number, total_round = para()
user_number_base = user_number
quitMode, quit_range = para_quit()

# Each client creates a random seed corresponding to the server
GlobalModel = VGG16()
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

    if quitMode == 1:
        quitNum = user_number * quit_range // 100
        quitId = random_int_list(0, user_number - 1, quitNum, 1)
        # print(quitId)

    time_avg_computation = 0
    time_avg_cryptographic = 0
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
                for d in range(54):
                    send_data = json.dumps(data[d], cls=NpEncoder)
                    size_communication += (len(send_data) / float(1024*1024))
                flag = 0

            time0 = time.perf_counter()
            for p in range(54):
                decode[p] += data[p] * (1 / (user_number - quitNum_e))
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
        model_dict['features.0.weight'] += torch.from_numpy(decode[0])
        model_dict['features.0.bias'] += torch.from_numpy(decode[1])
        model_dict['features.1.weight'] += torch.from_numpy(decode[2])
        model_dict['features.1.bias'] += torch.from_numpy(decode[3])
        model_dict['features.3.weight'] += torch.from_numpy(decode[4])
        model_dict['features.3.bias'] += torch.from_numpy(decode[5])
        model_dict['features.4.weight'] += torch.from_numpy(decode[6])
        model_dict['features.4.bias'] += torch.from_numpy(decode[7])

        model_dict['features.7.weight'] += torch.from_numpy(decode[8])
        model_dict['features.7.bias'] += torch.from_numpy(decode[9])
        model_dict['features.8.weight'] += torch.from_numpy(decode[10])
        model_dict['features.8.bias'] += torch.from_numpy(decode[11])
        model_dict['features.10.weight'] += torch.from_numpy(decode[12])
        model_dict['features.10.bias'] += torch.from_numpy(decode[13])
        model_dict['features.11.weight'] += torch.from_numpy(decode[14])
        model_dict['features.11.bias'] += torch.from_numpy(decode[15])

        model_dict['features.14.weight'] += torch.from_numpy(decode[16])
        model_dict['features.14.bias'] += torch.from_numpy(decode[17])
        model_dict['features.15.weight'] += torch.from_numpy(decode[18])
        model_dict['features.15.bias'] += torch.from_numpy(decode[19])
        model_dict['features.17.weight'] += torch.from_numpy(decode[20])
        model_dict['features.17.bias'] += torch.from_numpy(decode[21])
        model_dict['features.18.weight'] += torch.from_numpy(decode[22])
        model_dict['features.18.bias'] += torch.from_numpy(decode[23])

        model_dict['features.20.weight'] += torch.from_numpy(decode[24])
        model_dict['features.20.bias'] += torch.from_numpy(decode[25])
        model_dict['features.21.weight'] += torch.from_numpy(decode[26])
        model_dict['features.21.bias'] += torch.from_numpy(decode[27])
        model_dict['features.24.weight'] += torch.from_numpy(decode[28])
        model_dict['features.24.bias'] += torch.from_numpy(decode[29])
        model_dict['features.25.weight'] += torch.from_numpy(decode[30])
        model_dict['features.25.bias'] += torch.from_numpy(decode[31])

        model_dict['features.27.weight'] += torch.from_numpy(decode[32])
        model_dict['features.27.bias'] += torch.from_numpy(decode[33])
        model_dict['features.28.weight'] += torch.from_numpy(decode[34])
        model_dict['features.28.bias'] += torch.from_numpy(decode[35])
        model_dict['features.30.weight'] += torch.from_numpy(decode[36])
        model_dict['features.30.bias'] += torch.from_numpy(decode[37])
        model_dict['features.31.weight'] += torch.from_numpy(decode[38])
        model_dict['features.31.bias'] += torch.from_numpy(decode[39])

        model_dict['features.34.weight'] += torch.from_numpy(decode[40])
        model_dict['features.34.bias'] += torch.from_numpy(decode[41])
        model_dict['features.35.weight'] += torch.from_numpy(decode[42])
        model_dict['features.35.bias'] += torch.from_numpy(decode[43])
        model_dict['features.37.weight'] += torch.from_numpy(decode[44])
        model_dict['features.37.bias'] += torch.from_numpy(decode[45])
        model_dict['features.38.weight'] += torch.from_numpy(decode[46])
        model_dict['features.38.bias'] += torch.from_numpy(decode[47])

        model_dict['features.40.weight'] += torch.from_numpy(decode[48])
        model_dict['features.40.bias'] += torch.from_numpy(decode[49])
        model_dict['features.41.weight'] += torch.from_numpy(decode[50])
        model_dict['features.41.bias'] += torch.from_numpy(decode[51])
        model_dict['classifier.weight'] += torch.from_numpy(decode[52])
        model_dict['classifier.bias'] += torch.from_numpy(decode[53])

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
