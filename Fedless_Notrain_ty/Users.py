import binascii
import json
import random
import sys
import time
from io import BytesIO
import numpy as np

import torch

from LocalTrain import LocalTrain
from Initial import para, DataSize, Logger, para_quit, random_int_list, quit_random, addtwodimdict, NpEncoder, dimension

sys.stdout = Logger('log_user.txt')

user_number, total_round = para()
user_number_base = user_number
quitMode, quit_range = para_quit()

# Each client creates a random seed corresponding to the server
r, c = dimension()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global user_number, time0, time1, flag
    t_acc = np.zeros((total_round,))
    quitNum = 0
    quitId = []
    quitNum_e = 0
    quitId_e = []
    quit_seed = 6
    train_seed = 1
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
                quitId_new.append(quitId[i])
            quitNum_e += quit_user_num
            quit_seed += 1
        print('The number of users in round %d: %d' % (int(e + 1), user_number - quitNum_e))

        decode = np.zeros((r, c))

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

            data = LocalTrain(r, c, train_seed)

            if flag == 1:
                send_data = json.dumps(data, cls=NpEncoder)
                size_communication += (len(send_data) / float(1024 * 1024))
                flag = 0

            time0 = time.perf_counter()
            decode += data * (1 / (user_number - quitNum_e))
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

    print('+++++++++++')
    print('Average computational cost of a user per round: %f ms' % (time_avg_computation / total_round))
    print('Average communication cost of a user per round: %f MB' % (size_avg_communication / total_round))
    print('+++++++++++')


if __name__ == '__main__':
    main()
