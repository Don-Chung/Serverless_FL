import binascii
import json
import random
import sys
import time
from io import BytesIO
import numpy as np

import torch

from AccTest import test_Acc
from Gmask import get_secrets
from LocalTrain import LocalTrain
from Initial import para, DataSize, Logger, para_quit, random_int_list, quit_random, addtwodimdict, NpEncoder, dimension
from PySSSS import ss_encode, ss_decode
from aes import AES
from keyAgreement import private_keys, ka

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

    # key agreement
    sk = private_keys(user_number, 66)
    keys = ka(sk)

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

        secrets = {}
        masks = np.zeros((user_number,))
        n = user_number - len(quitId_e)
        # secrets->masks
        for j in range(user_number):
            if j in quitId_e:
                continue
            if flag == 1:
                time0 = time.perf_counter()
            secrets.setdefault(j, get_secrets(n, secert_seed))
            secert_seed += 1
            if flag == 1:
                time1 = time.perf_counter()
                time_computation += (time1 - time0)
                time_cryptographic += (time1 - time0)
                flag = 0
        flag = 1
        index = 0
        for j in range(user_number):
            if j in quitId_e:
                continue
            if flag == 1:
                time0 = time.perf_counter()
            for k in range(user_number):
                if k in quitId_e:
                    continue
                masks[j] = masks[j] + secrets[k][index]
            index += 1
            if flag == 1:
                time1 = time.perf_counter()
                time_computation += (time1 - time0)
                time_cryptographic += (time1 - time0)
                flag = 0
        flag = 1

        t = int(n / 2)
        masks_str = {}
        shares = dict()  # user j持有的user i的份额

        # ss.share
        for i in range(user_number):
            if i in quitId_e:
                continue
            if flag == 1:
                time0 = time.perf_counter()
            # masks_str.setdefault(i, str(masks[i]))
            # print(masks_str[i])
            mask_i = BytesIO(str(masks[i]).encode('UTF-8'))
            outputs = []
            for j in range(user_number):
                if j == i or j in quitId_e:
                    continue
                outputs.append(BytesIO())

            ss_encode(mask_i, outputs, t)
            index = 0
            for j in range(user_number):
                if j == i or j in quitId_e:
                    addtwodimdict(shares, j, i, '#')
                    continue
                addtwodimdict(shares, j, i, binascii.hexlify(outputs[index].getvalue()))
                index += 1
            if flag == 1:
                time1 = time.perf_counter()
                time_computation += (time1 - time0)
                time_cryptographic += (time1 - time0)
                flag = 0
        flag = 1

        for j in range(user_number):
            # Local Train
            if j == 0:
                print("users start training")
            if j in quitId_new or j in quitId_e:
                continue

            data = LocalTrain(r, c, train_seed)

            if flag == 1:
                time0 = time.perf_counter()
            data = data + masks[j]
            if flag == 1:
                time1 = time.perf_counter()
                time_computation += (time1 - time0)
                time_cryptographic += (time1 - time0)
                send_data = json.dumps(data, cls=NpEncoder)
                size_communication += (len(send_data) / float(1024 * 1024))
                flag = 0

            time0 = time.perf_counter()
            decode += data * (1 / (user_number - quitNum_e))
            time1 = time.perf_counter()
            time_computation += (time1 - time0)

        # flag = 1
        # # aes加密
        # for j in range(user_number):
        #     if j in quitId_e:
        #         continue
        #     if flag == 1:
        #         time0 = time.perf_counter()
        #     for i in range(user_number):
        #         if j == i or i in quitId_e:
        #             continue
        #         for k in quitId_new:
        #             master = str(keys[j][i]).encode()
        #             ase = AES(master)
        #             shares[j][k] = ase.Encrypt(shares[j][k])
        #             shares[j][k] = ase.Decrypt(shares[j][k])
        #             # print(shares[j][k])
        #     if flag == 1:
        #         time1 = time.perf_counter()
        #         flag = 0
        #         time_computation += (time1 - time0)
        #         time_cryptographic += (time1 - time0)
        #
        # for j in range(user_number):
        #     if j in quitId_e:
        #         continue
        #     for i in range(user_number):
        #         if j == i or i in quitId_e:
        #             continue
        #         for k in quitId_new:
        #             size_communication += (len(shares[j][k]) / float(1024*1024))
        #     break
        #
        # flag = 1
        # # aes解密
        # for j in range(user_number):
        #     if j in quitId_e:
        #         continue
        #     if flag == 1:
        #         time0 = time.perf_counter()
        #     for i in range(user_number):
        #         if j == i or i in quitId_e:
        #             continue
        #         for k in quitId_new:
        #             master = str(keys[j][i]).encode()
        #             ase = AES(master)
        #             shares[j][k] = ase.Decrypt(shares[j][k])
        #             # print(shares[j][k])
        #     if flag == 1:
        #         time1 = time.perf_counter()
        #         flag = 0
        #         time_computation += (time1 - time0)
        #         time_cryptographic += (time1 - time0)

        # ss.recon
        time0 = time.perf_counter()
        for k in quitId_new:
            inputs = []
            index = 0
            for j in range(user_number):
                if j in quitId_new or j in quitId_e:
                    continue
                inputs.append(BytesIO(binascii.unhexlify(shares[j][k])))
                inputs[index].seek(0)
                index += 1
            output = BytesIO()
            ss_decode(inputs, output)
            mask = float(output.getvalue().decode('UTF-8'))
            send_mask = json.dumps(mask, cls=NpEncoder)
            size_communication += (len(send_mask) / float(1024 * 1024))

            decode += mask * (1 / (user_number - quitNum_e))
        time1 = time.perf_counter()
        # print('toll: %f' % (1000*(time1 - time0)))
        time_cryptographic += (time1 - time0)
        time_computation += (time1 - time0)
        print('Global model generated')

        for i in range(len(quitId_new)):
            quitId_e.append(quitId_new[i])

        time_computation = time_computation * 1000
        time_cryptographic = time_cryptographic * 1000
        time_avg_computation += time_computation
        time_avg_cryptographic += time_cryptographic
        size_avg_communication += size_communication
        print('Computational cost per user: %f ms' % time_computation)
        print('Cryptographic cost per user: %f ms' % time_cryptographic)
        print('Communication cost per user: %f MB' % size_communication)

    print('+++++++++++')
    print('Average computational cost of a user per round: %f ms' % (time_avg_computation / total_round))
    print('Average cryptographic cost of a user per round: %f ms' % (time_avg_cryptographic / total_round))
    print('Average communication cost of a user per round: %f MB' % (size_avg_communication / total_round))
    print('+++++++++++')


if __name__ == '__main__':
    main()
