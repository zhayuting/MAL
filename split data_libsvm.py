import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import KFold
from scipy.sparse import vstack
import os

# 加载数据集
file_path = 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/SUSY/SUSY_train.txt'
X, y = load_svmlight_file(file_path)

# 确保输出目录存在
output_dir = 'C:/Users/Administrator/PycharmProjects/DepthCompare/data/classify/SUSY'
os.makedirs(output_dir, exist_ok=True)

# 创建KFold对象，n_splits设为10
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化计数器
fold_idx = 1

# 遍历每个分割
for train_index, test_index in kf.split(X):
    # 获取当前分割的训练和测试数据
    X_fold, y_fold = X[test_index], y[test_index]

    # 将当前分割的数据保存到文件中
    fold_path = os.path.join(output_dir, f'SUSY_{fold_idx}.libsvm')
    with open(fold_path, 'w') as f:
        for i in range(X_fold.shape[0]):
            features = " ".join([f"{j + 1}:{X_fold[i, j]}" for j in range(X_fold.shape[1])])
            f.write(f"{y_fold[i]} {features}\n")

    # 更新计数器
    fold_idx += 1

print("数据集已均匀分割并保存到文件中。")
