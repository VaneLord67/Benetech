import json
import os
import random
import shutil

import cv2


def split_dataset():
    # 原始数据集文件夹路径
    dataset_dir = './dataset/train/vertical_bar/images'
    # 划分后的训练集和测试集文件夹路径
    train_dir = './datasets/vertical_bar/images/train'
    test_dir = './datasets/vertical_bar/images/val'
    # 训练集占比
    train_ratio = 0.8

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

        id, _ = os.path.splitext(filename)
        write(id, f'./datasets/vertical_bar/labels/train/{id}.txt')

    # 将剩余的文件复制到测试集文件夹
    for filename in filenames[split_index:]:
        src_path = os.path.join(dataset_dir, filename)
        dst_path = os.path.join(test_dir, filename)
        shutil.copy(src_path, dst_path)

        id, _ = os.path.splitext(filename)
        write(id, f'./datasets/vertical_bar/labels/val/{id}.txt')


def write(id, path):
    img = cv2.imread(f'./dataset/train/vertical_bar/images/{id}.jpg')
    img_height, img_width, _ = img.shape
    # 打开 JSON 文件
    with open(f'./dataset/train/vertical_bar/annotations/{id}.json', 'r') as f:
        # 读取 JSON 数据
        data = json.load(f)
    # 获取 "bars" 列表
    bars = data["visual-elements"]["bars"]
    labels = []
    for bar in bars:
        bar_height = bar["height"]
        bar_width = bar["width"]
        x0 = bar["x0"]
        y0 = bar["y0"]
        center_x = x0 + bar_width / 2
        center_y = y0 + bar_height / 2
        if center_x / img_width > 1 or center_y / img_height > 1 or bar_width / img_width > 1 or bar_height / img_height > 1:
            continue
        label = f'0 {center_x / img_width} {center_y / img_height} {bar_width / img_width} {bar_height / img_height}'
        labels.append(label)
    with open(path, 'w') as f:
        for label in labels:
            f.write(label + "\n")


if __name__ == '__main__':
    split_dataset()