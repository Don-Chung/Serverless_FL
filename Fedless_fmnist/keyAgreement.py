import random

from Initial import addtwodimdict


def judge_prime(number):
    status = 0
    if number < 1 or type(number) != int:
        return 'type error'
    else:
        for i in range(2, number):
            if number % i == 0:
                status = 1
        return status


def isPrime(a, b):
    while b != 0:
        temp = b
        b = a % b
        a = temp
    if a == 1:
        return 1
    else:
        return 0


def find_prine(number):
    primeList = []
    for i in range(1, number + 1):
        ans = isPrime(i, number)
        if ans == 1:
            primeList.append(i)
    byg = []
    prime_temp = []
    for j in primeList:
        for i in range(1, len(primeList) + 1):
            prime_temp.append(j ** i % number)
        prime_temp.sort()
        if primeList == prime_temp:
            byg.append(j)
        else:
            pass
        prime_temp = []
    return byg


def power_func(base, exp, mod):
    if exp < 0:
        x = 1 / base
        y = -exp
    elif exp == 0:
        x = 1
    else:
        n = power_func(base, exp // 2, mod)
        x = n * n
        if exp % 2 == 1:
            x *= base
    return x % mod


p = 10000967
g = 2


def ka(sk):
    pk = []
    for i in sk:
        pk.append(power_func(g, i, p))
    keys = dict()
    for i in range(len(sk)):
        for j in range(len(pk)):
            addtwodimdict(keys, i, j, power_func(pk[j], sk[i], p))
    return keys


def private_keys(user_number, seed):
    sk = []
    for i in range(user_number):
        sk.append(random.randint(1000, 6000))
        random.seed(seed)
        seed += 1
    return sk
