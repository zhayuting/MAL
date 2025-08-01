import time

import numpy as np
import random


def generate_number_with_probability(p,q,k):
    """
    根据给定的概率 p 和数字列表生成一个随机数。

    :param p: 数字等于 numbers[0] 的概率
    :param numbers: 包含所有数字的列表，其中 numbers[0] 是特定数字，其他数字的概率均为 (1-p)/(len(numbers)-1)
    :return: 根据概率生成的数字
    """

    if not 0 <= p <= 1:
        raise ValueError("Probability p must be between 0 and 1.")

    # 计算其他位置的概率
    other_probability = (1 - p) / (q - 1)

    # 创建概率列表
    probabilities = [other_probability] * q
    probabilities[k] = p


    # 使用 random.choices 按照指定的概率权重进行选择
    numbers = [i for i in range(0, q)]
    chosen_number = random.choices(numbers, probabilities)[0]
    return chosen_number



# 对每个子集的处理
def EVTransf(matrix, X_train, d, n, q, p):  # matrix为分桶方式，d为特征数，n为样本数,q为分桶数
    # 每个样本生成一个规模为dq+1的随机向量
    A = np.random.rand(n, d * q + 1)
    # 按分桶构造矩阵方程
    X_train = X_train.astype('float64')
    B = np.zeros((d * q, d * q + 1))

    for i in range(0, d):  # features
        for k in range(0, q):  # buckets
            column = X_train.iloc[:, i]
            lower_bound = matrix[k, i]
            upper_bound = matrix[k + 1, i]
            #
            bool_indices = (column >= lower_bound) & (column <= upper_bound) if k == 0 else (column > lower_bound) & (
                    column <= upper_bound)
            # 获取符合条件的数值的索引
            indices = np.where(bool_indices)[0]

            # 修改X_train index为indices的值中前num_k个值 为桶中值
            column.loc[indices] = 0.5 * (lower_bound + upper_bound)
            B[k + q * i, :] = np.sum(A[indices, :], axis=0)

    return X_train