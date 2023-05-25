import json
import os
import random
import shutil

import pandas as pd

import graph_data_loader
import read_result


def split_dataset():
    # 原始数据集文件夹路径
    dataset_dir = 'dataset/train/images'
    # 划分后的训练集和测试集文件夹路径
    train_dir = 'huggingface/dataset/train'
    test_dir = 'huggingface/dataset/test'
    # 训练集占比
    train_ratio = 0.05
    # 获取数据集中所有文件名
    filenames = os.listdir(dataset_dir)
    # 随机打乱文件名顺序
    random.shuffle(filenames)
    # 计算训练集和测试集的分界点
    split_index = int(len(filenames) * train_ratio)
    # 将前 split_index 个文件复制到训练集文件夹
    for filename in filenames[:split_index]:
        src_path = os.path.join(dataset_dir, filename)
        dst_path = os.path.join(train_dir, filename)
        shutil.copy(src_path, dst_path)
    # # 将剩余的文件复制到测试集文件夹
    # for filename in filenames[split_index:]:
    #     src_path = os.path.join(dataset_dir, filename)
    #     dst_path = os.path.join(test_dir, filename)
    #     shutil.copy(src_path, dst_path)


def write():
    csv_file_path = "huggingface/dataset/train/metadata.csv"
    header = ['file_name', 'text']
    df = pd.DataFrame(columns=header)
    df.to_csv(csv_file_path, header=True, index=False)
    gdl = graph_data_loader.GraphDataLoader()
    file_generator = gdl.list_files("huggingface/dataset/train")
    for filepath in file_generator:
        id = read_result.ReadResult.get_id(filepath)
        if id == 'metadata':
            continue
        text = "<|BOS|>"
        with open(f'dataset/train/annotations/{id}.json', 'r') as f:
            data = json.load(f)
            data_series = data['data-series']
            chart_type = data['chart-type']
            text += f'<{chart_type}><x_start>'
            for idx, d in enumerate(data_series):
                x = d['x']
                if isinstance(x, str):
                    x = x.replace("\n", "")
                text += f'{x}' + (";" if idx != len(data_series) - 1 else "")
            text += "<x_end><y_start>"
            for idx, d in enumerate(data_series):
                y = d['y']
                text += f'{y}' + (";" if idx != len(data_series) - 1 else "")
            text += "<y_end>"
        df = pd.DataFrame({
            'file_name': id + ".jpg",
            'text': text,
        }, index=[0])
        df.to_csv(csv_file_path, index=False, header=False, mode='a')


if __name__ == '__main__':
    split_dataset()
    write()
