# -*- coding: utf-8 -*-

"""
Created on Mon Nov 25 18:08:12 2019

@author: tuan
"""
import h5py

# 打开 h5 文件
file_path = '/home/cwh/Desktop/cnntest/ur_train_dataset_BoxBlue.h5'
with h5py.File(file_path, 'r') as file:
    # 打印文件中所有的键
    print("Keys:", list(file.keys()))

    # 获取具体数据集
    dataset_key = 'datum'  # 这里的键名可能需要根据文件结构进行调整
    dataset_group = file[dataset_key]

    # 打印数据集的相关信息
    #print("Dataset Info:")
    #print("Group Keys:", list(dataset_group.keys()))

    # 选择一个数据集
    subset_key = '0014134' # 根据实际情况替换为正确的子集键名
    subset_dataset = dataset_group[subset_key]

    # 打印子集数据集的相关信息
    print("Subset Dataset Info:")
    print("Shape:", subset_dataset.shape)
    print("Dtype:", subset_dataset.dtype)

    # 读取数据集的内容
    data = subset_dataset[()]

    # 打印数据
    print("Data:", data)

