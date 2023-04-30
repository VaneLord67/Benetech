import json
import os
import shutil


def get_chart_type_from_json(file_path):
    """
    # 定义函数来解析json文件
    """
    with open(file_path) as f:
        data = json.load(f)
        chart_type = data.get("chart-type", None)
        return chart_type


# 定义函数来遍历目录
def traverse_directory(directory):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                chart_type = get_chart_type_from_json(file_path)
                yield filename, chart_type


if __name__ == '__main__':
    directory = "./dataset/train/annotations"
    for filename, chart_type in traverse_directory(directory):
        print(filename, chart_type)
        src_file_jpg = "./dataset/train/images/" + filename[:-5] + ".jpg"
        src_file_annotation = "./dataset/train/annotations/" + filename[:-5] + ".json"
        dest_file = "./dataset/train/" + chart_type + "/"
        dest_file_jpg = dest_file + "images/" + filename[:-5] + ".jpg"
        dest_file_annotation = dest_file + "annotations/" + filename[:-5] + ".json"
        shutil.copyfile(src_file_jpg, dest_file_jpg)
        shutil.copyfile(src_file_annotation, dest_file_annotation)

