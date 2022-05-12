import math
from queue import Queue

import numpy as np

from Initial import para
import PBtree


def get_secrets(user_number, seed):
    # user_number, total_round = para()
    n = math.ceil(np.log2(user_number))
    lastSecondNumber = 2 ** (n - 1)
    lastNumber = user_number - lastSecondNumber
    lastNumberTmp = 0
    depth = 0
    root = 0.0

    tree = PBtree.PerfectBinaryTree()
    queue = Queue()
    queue.put(root)

    while not queue.empty():
        size = queue.qsize()
        depth += 1
        for j in range(size):
            cur = queue.get()
            tree.append(cur)
            if depth > n or lastNumberTmp >= lastNumber:
                continue
            r = PBtree.g_rand(seed)
            seed += 1
            queue.put(cur + r)
            queue.put(-r)
            if depth == n:  # 最后一层
                lastNumberTmp += 1
    return tree.leaf()
# tree.show()
# leafs = tree.leaf()
# print(leafs)
# ss = 0.0
# for i in range(len(leafs)):
#     ss += leafs[i]
# print(ss)
