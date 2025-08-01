import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.datasets import load_svmlight_file
import csv
import numpy as np

file_path = 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/regression/ca_data/cadata_train.t'

X_train, y_train = load_svmlight_file(file_path)
file_path_1 = 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/regression/ca_data/cadata_test.t'
X_test, y_test = load_svmlight_file(file_path)

file_train_cls_path = {
    # 训练 分类 数据集
    'credit':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/regression/Credit_1/',  # cs_data_split_1.csv
    'cadata':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/regression/ca_data/',  # cadata.libsvm
    'a6a': 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/a6a/',  # a6a_1.libsvm
    'a7a':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/a7a/',  # a7a_1.libsvm
    'uci':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/UCI_data/',  # UCI_Credit_Card.csv
    'cod':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/cod-rna/',  # cod-rna.libsvm
    'diabetes':'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/diabetes/',  # diabetes.libsvm
}


def preprocess_data(dataset_name, file_path):
    if dataset_name == 'credit':

        file_name = f'{file_path}cs-training.csv'
        # 读取文件，定义变量（这里假设读取的是某种数据）
        data = pd.read_csv(file_name)

        X_train = data.drop(columns=['id', 'SeriousDlqin2yrs'])
        y_train = data['SeriousDlqin2yrs']
        file_path_1 = f'{file_path}cs-test.csv'
        data_1 = pd.read_csv(file_path_1)
        file_path_2 = f'{file_path}sampleEntry.csv'
        data_2 = pd.read_csv(file_path_2)
        X_test = data_1.drop(columns=['id', 'SeriousDlqin2yrs'])
        y_test = data_2['Probability']

        return X_train, y_train, X_test, y_test

    elif dataset_name == 'cadata':
        prefix_test = 'cadata_test'
        file_name = f'{file_path}/cadata_train.t'

        # if dataset_name == 'a6a':
        #     file_name = f'{file_path}/a6a.txt'
        # elif dataset_name == 'deabetes':
        #     file_name = f'{file_path}/diabetes_train.txt'
        # else:
        #     file_name = f'{file_path}/cod-rna_train.txt'

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
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',# 目标函数
        'tree_method': 'hist',            # 使用直方图算法
        'max_depth': 1,  # 树的最大深度
        'eta': 0.01,  # 学习率
        # 'subsample': 0.8,                 # 子样本比例
        'colsample_bytree': 0.9,
        # 'max_bin':32,# 每棵树的特征采样比例
    }

    # 训练模型
    num_boost_round = T
    num_boost_round = T
    bst = xgb.train(params, dtrain, num_boost_round)

    # 预测
    y_pred = bst.predict(dtest)

    # 评估模型
    mse = mean_squared_error(y_test_total, y_pred)
    rmse = np.sqrt(mse)
    y_mean = np.mean(y_test_total)
    # 将 MSE 转换为相对于平均值的百分比
    mse_percentage = (mse / y_mean ** 2) * 100
    print(f'mse_percentage Error: {mse_percentage}')
    return [dataset_name, T, mse_percentage]

def write_log(results):
    csv_filename = 'results reg_q40.csv'
    fieldnames = ['dataset', 'T', 'mse_percentage']
    # 写入CSV文件
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # 只有第一次写入时需要写入头部
        if file.tell() == 0:  # 文件开始位置为0，表示是新文件
            writer.writeheader()
        writer.writerow({fieldnames[0]: results[0], fieldnames[1]: results[1], fieldnames[2]: results[2]})


if __name__ == '__main__':
    for dataset_name in ['cadata']:
        print(f'Processing dataset: {dataset_name}')
        for T in [1000]:
            print(f'T = {T}')
            # params['max_depth'] = 1
            results = train(dataset_name)
            # write_log(results)



# data = pd.read_csv(file_path)
# X_train = data.drop(columns=['id', 'SeriousDlqin2yrs'])
# y_train = data['SeriousDlqin2yrs']
# file_path_1 = 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/regression/Credit_1/cs-test.csv'
# data_1 = pd.read_csv(file_path_1)
# file_path_2 = 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/regression/Credit_1/sampleEntry.csv'
# data_2 = pd.read_csv(file_path_2)
# X_test = data_1.drop(columns=['id', 'SeriousDlqin2yrs'])
# y_test = data_2['Probability']

"""
# 假设第一列为 'id' 列，目标变量列名为 'target'，其余列为特征
X = data.drop(columns=['ID', 'default.payment.next.month'])
y = data['default.payment.next.month']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=42
"""

# # 将数据转换为DMatrix格式，这是XGBoost的特定数据格式
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
#
# # 设置参数，使用直方图算法
# params = {
#     'objective': 'reg:squarederror',  # 目标函数
#     #'tree_method': 'hist',            # 使用直方图算法
#     'max_depth': 1,                   # 树的最大深度
#     'eta': 0.005,                       # 学习率
#     #'subsample': 0.8,                 # 子样本比例
#     'colsample_bytree': 0.9,
#     #'max_bin':30,# 每棵树的特征采样比例
# }
#
# # 训练模型
# for T in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]:
#     print(f'T = {T}')
#     num_boost_round = T
#     bst = xgb.train(params, dtrain, num_boost_round)
#
#     # 预测
#     y_pred = bst.predict(dtest)
#
#     # 评估模型
#     mse = mean_squared_error(y_test, y_pred)
#     print(f'Mean Squared Error: {mse}')
# num_boost_round = 100
# bst = xgb.train(params, dtrain, num_boost_round)
#
# # 预测
# y_pred = bst.predict(dtest)
#
# # 评估模型
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
