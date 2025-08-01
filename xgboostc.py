import time

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file
import numpy as np
import csv

# file_path = 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/cod-rna/cod-rna_train.txt'
# X_train, y_train = load_svmlight_file(file_path)
# file_path = 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/cod-rna/cod-rna_test.t'
# X_test, y_test = load_svmlight_file(file_path)
# y_train[y_train == -1] = 0
# y_test[y_test == -1] = 0

# data = pd.read_csv(file_path)
#
# # 2. 数据预处理
# # 假设第一列为 'id' 列，目标变量列名为 'target'，其余列为特征
# X = data.drop(columns=['ID', 'default.payment.next.month'])
# y = data['default.payment.next.month']
#
# # 分割数据集为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=42

file_train_cls_path = {
    # 训练 分类 数据集
    'credit':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/regression/Credit_1/',  # cs_data_split_1.csv
    'cadata':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/regression/ca_data/',  # cadata.libsvm
    'a6a': 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/a6a/',  # a6a_1.libsvm
    'a7a':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/a7a/',  # a7a_1.libsvm
    'uci':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/UCI_data/',  # UCI_Credit_Card.csv
    'cod':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/cod-rna/',  # cod-rna.libsvm
    'diabetes':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/diabetes/',  # diabetes.libsvm
    'SUSY_':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/SUSY_/',
}

shape_dataset = {
    'credit': (10, 15000),
    'cadata': (20640, 8),
    'a6a': (122,1122),
    'a7a': (123, 1122),
    'uci': (23, 2400),
    'cod': (8, 5953),
    'diabetes': (8, 61),
    'SUSY_': (18, 23146),
}

def preprocess_data(dataset_name, file_path):
    if dataset_name in ('credit', 'uci'):
        prefix = 'credit_' if dataset_name == 'credit' else 'uci_'
        file_name = f'{file_path}{prefix}train.csv'
        # 读取文件，定义变量（这里假设读取的是某种数据）
        data = pd.read_csv(file_name)
        if dataset_name == 'uci':
            X_train = data.drop(columns=['ID', 'default.payment.next.month'])  ################################
            y_train = data['default.payment.next.month']  #############################################
            file_name1 = f'{file_path}uci_test.csv'
            # 读取文件，定义变量（这里假设读取的是某种数据）
            data1 = pd.read_csv(file_name1)
            X_test = data1.drop(columns=['ID', 'default.payment.next.month'])  ################################
            y_test = data1['default.payment.next.month']

        else:
            X_train = data.drop(columns=['id', 'SeriousDlqin2yrs'])
            y_train = data['SeriousDlqin2yrs']
            file_path_1 = f'{file_path}credit_test.csv'
            data_1 = pd.read_csv(file_path_1)
            X_test = data_1.drop(columns=['id', 'SeriousDlqin2yrs'])
            y_test = data_1['SeriousDlqin2yrs']

        return X_train, y_train, X_test, y_test

    elif dataset_name in ('a6a' , 'cod', 'diabetes', 'SUSY_'):
        prefix_test = 'SUSY__test' if dataset_name == 'SUSY_' else 'diabetes_test' if dataset_name == 'diabetes' else 'cod-rna_test' if dataset_name == 'cod' else 'a6a_test'

        if dataset_name == 'a6a':
            file_name = f'{file_path}/a6a.txt'
        elif dataset_name == 'diabetes':
            file_name = f'{file_path}/diabetes_train.txt'
        elif dataset_name == 'SUSY_':
            file_name = f'{file_path}/SUSY__train.txt'
        else:
            file_name = f'{file_path}/cod-rna_train.txt'

            # 读取文件，定义变量（这里假设读取的是某种数据）
        X_train, y_train = load_svmlight_file(file_name)
        y_train[y_train == -1] = 0  # a6a a7a中将标签的-1替换为0

        file_path_11 = f'{file_path}/{prefix_test}.t'
        X_test, y_test = load_svmlight_file(file_path_11)
        X_test = pd.DataFrame(X_test.toarray())
        y_test[y_test == -1] = 0  # a6a a7a中将标签的-1替换为0
        return X_train, y_train, X_test, y_test

    else:
        raise ValueError("Unsupported dataset name")

    return processed_data

def train(dataset_name):
    # load data
    # dataset_name = 'dataset_name'
    X_train_total, y_train_total, X_test_total, y_test_total = preprocess_data(dataset_name, file_train_cls_path[dataset_name])

    # 合并多个X_train和y_train   已经在预处理中实现
    # X_train_total,y_train_total = pd.concat(X_train_total),np.concatenate(y_train_total)

    # 将数据转换为DMatrix格式，这是XGBoost的特定数据格式
    dtrain = xgb.DMatrix(X_train_total, label=y_train_total)
    dtest = xgb.DMatrix(X_test_total, label=y_test_total)

    # 设置参数，使用直方图算法
    params = {
        'objective': 'reg:squarederror',  # 目标函数binary:logistic
        'tree_method': 'hist',  # 使用直方图算法
        # 'max_depth': 1,  # 树的最大深度
        'eta': 0.1,  # 学习率
        # 'subsample': 0.8,                 # 子样本比例
        'colsample_bytree': 0.9,  # 每棵树的特征采样比例
        'max_bin': 24,

    }

    # 训练模型
    num_boost_round = T
    start_time = time.time()
    bst = xgb.train(params, dtrain, num_boost_round)
    end_time = time.time()
    Training_time = end_time - start_time
    print(f'Training time: {Training_time}')

    # 预测
    y_pred = bst.predict(dtest)

    # 将预测结果转换为二进制标签
    y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]

    # 计算准确率
    accuracy = accuracy_score(y_test_total, y_pred_binary)
    print(f"Accuracy: {accuracy}")

    # 评估模型
    mse = mean_squared_error(y_test_total, y_pred_binary)
    print(f'Mean Squared Error: {mse}')

    auc = roc_auc_score(y_test_total, y_pred_binary)
    print('AUC:', auc)
    return [dataset_name, T, auc, Training_time]

def write_log(results):
    csv_filename = 'depth.csv'
    fieldnames = ['dataset', 'T', 'AUC', 'Training_time']
    # 写入CSV文件
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # 只有第一次写入时需要写入头部
        if file.tell() == 0:  # 文件开始位置为0，表示是新文件
            writer.writeheader()
        writer.writerow({fieldnames[0]: results[0], fieldnames[1]: results[1], fieldnames[2]: results[2], fieldnames[3]: results[3]})


if __name__ == '__main__':
    for dataset_name in ['SUSY_']:
        print(f'Processing dataset: {dataset_name}')
        # for T in [50,100,200,300,400,500,600,700,800,900,1000]:
        # T = 1000
        # print(f'T = {T}')
        # params['max_depth'] = 1
        # st = time.time()
        T=500
        results = train(dataset_name)
        # et=time.time()
        # print(f'Elapsed time: {et-st}')
        #     write_log(results)

# # 将数据转换为DMatrix格式，这是XGBoost的特定数据格式
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
#
# # 设置参数，使用直方图算法
# params = {
#     'objective': 'binary:logistic',
#     #'tree_method': 'hist',            # 使用直方图算法
#     'max_depth': 1,                   # 树的最大深度
#     'eta': 0.005,                       # 学习率
#     #'subsample': 0.8,                 # 子样本比例
#     'colsample_bytree': 0.9,          # 每棵树的特征采样比例
#     #'max_bin':30,
# }
#
# # 训练模型
# num_boost_round = 15000
# bst = xgb.train(params, dtrain, num_boost_round)
#
# # 预测
# y_pred = bst.predict(dtest)
#
# # 将预测结果转换为二进制标签
# y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]
#
# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred_binary)
# print(f"Accuracy: {accuracy}")
#
# # 评估模型
# mse = mean_squared_error(y_test, y_pred_binary)
# print(f'Mean Squared Error: {mse}')
#
# auc = roc_auc_score(y_test, y_pred_binary)
# print('AUC:', auc)
