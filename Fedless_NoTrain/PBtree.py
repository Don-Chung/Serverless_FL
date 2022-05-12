# coding=utf-8
import random

from array import array


class Node(object):
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.left_child = None
        self.right_child = None


def g_rand(seed):
    random.seed(seed)
    return random.random()


class TreeQueue(object):
    def __init__(self):
        self.__members = list()

    def is_empty(self):
        return not len(self.__members)

    def enter(self, data):
        self.__members.insert(0, data)

    def outer(self):
        if self.is_empty():
            return
        return self.__members.pop()


class PerfectBinaryTree(object):

    def __init__(self):
        self.__root = None

    def is_empty(self):
        return not self.__root

    def append(self, data):
        node = Node(data)
        if self.is_empty():
            self.__root = node
            return
        queue = TreeQueue()
        queue.enter(self.__root)
        while not queue.is_empty():
            cur = queue.outer()
            if cur.left_child is None:
                cur.left_child = node
                node.parent = cur
                return
            queue.enter(cur.left_child)
            if cur.right_child is None:
                cur.right_child = node
                node.parent = cur
                return
            queue.enter(cur.right_child)

    def show(self):
        if self.is_empty():
            print('空二叉树')
            return
        queue = TreeQueue()
        queue.enter(self.__root)
        while not queue.is_empty():
            cur = queue.outer()
            print(cur.data, end=' ')
            if cur.left_child is not None:
                queue.enter(cur.left_child)
            if cur.right_child is not None:
                queue.enter(cur.right_child)
        print()

    def leaf(self):
        if self.is_empty():
            print('空二叉树')
            return
        queue = TreeQueue()
        queue.enter(self.__root)
        leafs = {}
        index = 0
        while not queue.is_empty():
            cur = queue.outer()
            if cur.left_child is None and cur.right_child is None:
                leafs.setdefault(index, cur.data)
                index += 1
            if cur.left_child is not None:
                queue.enter(cur.left_child)
            if cur.right_child is not None:
                queue.enter(cur.right_child)
        return leafs
