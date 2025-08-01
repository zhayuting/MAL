import time

import numpy as np
import pandas as pd
from scipy.linalg import null_space

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

def subsethandling_vector(matrix, X_train, y_train, d, n, q, p):
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

            # k_new = np.where(column[indices],generate_number_with_probability(p, q, k),k)
            # # 统计仍然分在k中的数量
            #
            #
            #
            # num_k = np.sum(k_new==k)
            # # 删除k 得到剩下的其他桶元素
            # last_k_new = np.delete(k_new, np.where(k_new == k))

            ########################################################################################
            num = len(indices)
            num_k = int(num * p)
            num_each_other = (num - num_k)// (q - 1)
            last_k_new = list(range(q))
            # 删除元素 k
            last_k_new = [i for i in last_k_new if i != k]

            # 修改X_train index为indices的值中前num_k个值 为桶中值
            column.loc[indices[:num_k]] = 0.5 * (lower_bound + upper_bound)
            B[k + q * i, :] = np.sum(A[indices, :][:num_k], axis=0)

            if p != 1:
                for index,err_k in np.ndenumerate(last_k_new):
                    lower_bound1 = matrix[err_k, i]
                    upper_bound1 = matrix[err_k + 1, i]
                    start_idx = num_k + index[0] * num_each_other
                    end_idx = num_k + (index[0] + 1) * num_each_other
                    column.loc[indices[start_idx : end_idx]] = 0.5 * (lower_bound1 + upper_bound1)
                    B[err_k + q * i, :] = np.sum(A[indices, :][start_idx : end_idx], axis=0)
            X_train.iloc[indices, i] = column.iloc[indices]

            # for index,err_k in np.ndenumerate(last_k_new):
            #     lower_bound1 = matrix[err_k, i]
            #     upper_bound1 = matrix[err_k + 1, i]
            #     column.loc[indices[num_k+index]] = 0.5 * (lower_bound1 + upper_bound1)
            #     B[err_k + q * i, :] = np.sum(A[indices, :][num_k+index], axis=0)
            # X_train.iloc[indices, i] = column[indices]
            # 还剩 len(unique_k_new) 个值,将这些值分到其他桶k_new中
            # 对于分到k中的每一个值
            # B[k + q * i, :] = np.sum(A[indices, :])
    # 解矩阵方程B
    # 方法 1：使用 SVD 解齐次方程

    U, S, VT = np.linalg.svd(B)
    X_svd = VT.T[:, -1]

    y_train = y_train.astype('float64')

    # 为标签加掩码
    for i in range(0, n):
        y_train[i] += np.dot(A[i, :], X_svd)

    return y_train,X_train


# 对每个子集的处理
def subsethandling(matrix, X_train, y_train, d, n, q, p):  # matrix为分桶方式，d为特征数，n为样本数,q为分桶数
    # 每个样本生成一个规模为dq+1的随机向量
    A = np.zeros((n, d * q + 1))
    #A = pd.zeros((n, d * q + 1))
    for i in range(0, n):
        for j in range(0, d * q + 1):
            A[i, j] = np.random.rand()
    # 按分桶构造矩阵方程
    X_train = X_train.astype('float64')
    B = np.zeros((d * q, d * q + 1))
    for i in range(0, d):  # features
        for k in range(0, q):  # buckets
            column = X_train.iloc[:, i]
            lower_bound = matrix[k, i]
            upper_bound = matrix[k+1, i]
            #
            bool_indices = (column >= lower_bound) & (column <= upper_bound) if k==0 else (column > lower_bound) & (column <= upper_bound)
            # 获取符合条件的数值的索引
            indices = np.where(bool_indices)[0]
            for j in indices:
                k = generate_number_with_probability(p, q, k)
                lower_bound1 = matrix[k, i]
                upper_bound1 = matrix[k + 1, i]
                B[k + q * i, :] += A[j, :]
                #if k != q-1 and k != 0:
                X_train.iloc[j, i] = 0.5*(lower_bound1+upper_bound1)
                # elif k == 0:
                #     X_train.iloc[j, i] = lower_bound
                # else:
                #     X_train.iloc[j, i] = upper_bound

        """
        for j in range(0, n):  # samples
            for k in range(0, q): # buckets
                if X_train.iloc[j, i] > matrix[k, i] and X_train.iloc[j, i] <= matrix[k + 1, i]:
                    B[k + q * i, :] += A[j, :]
                    if k != 0 :
                        X_train.iloc[j, i] = matrix[k + 1, i]
                        #X_train.iloc[j, i] = 0.5 * (matrix[k, i] + matrix[k + 1, i])
                    else:
                        X_train.iloc[j, i] = matrix[k, i]
        

                
                if k != q-1:
                    
                    if X_train.iloc[j, i] > matrix[k, i] and X_train.iloc[j, i] <= matrix[k+1, i]:
                        B[k, :] += A[j, :]
                        X_train.iloc[j, i]=1/2*(matrix[k, i]+matrix[k+1, i])
                else:
                    if X_train.iloc[j, i] > matrix[k, i] :
                        B[k, :] += A[j, :]
                        X_train.iloc[j, i]=matrix[k, i]
        """

    # 解矩阵方程B
    # 方法 1：使用 SVD 解齐次方程

    U, S, VT = np.linalg.svd(B)
    X_svd = VT.T[:, -1]

    y_train = y_train.astype('float64')

    # 为标签加掩码
    for i in range(0, n):
        y_train[i] += np.dot(A[i, :], X_svd)

    return y_train,X_train

