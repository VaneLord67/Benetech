from abstract_graph_reader import AbstractGraphReader
from read_result import ReadResult
import math
from ultralytics import YOLO
import os
import numpy as np
import cv2
import easyocr
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
from graph_classifier.graph_classifier_lenet import GraphType
from env import *


def take_first_element(a):
    return a[0]


# 由于坐标轴上数据全部都为等差数列，此处通过将数据化为等差数列，将坐标轴数据矫正
def make_linear(num_strings, n_0, n_0_w, flag):
    d_list = []
    d = 0
    dealt_strings = []
    for j in range(len(num_strings) - 1):
        d_list.append(abs(num_strings[j + 1] - num_strings[j]))

    d = find_most(d_list)
    if flag == 0:
        for i in range(len(num_strings)):
            if num_strings[i] == 0.0:
                dealt_strings.append(abs(n_0 - d * (n_0_w - i)))
            else:
                if num_strings[i] == abs(n_0 - d * (n_0_w - i)) or num_strings[i] == num_strings[i - 1] + d \
                        or num_strings[i] == num_strings[i + 1] - d:
                    dealt_strings.append(num_strings[i])
                else:
                    dealt_strings.append(abs(n_0 - d * (n_0_w - i)))
    else:
        for i in range(len(num_strings)):
            if num_strings[i] == 0.0:
                dealt_strings.append(abs(n_0 + d * (n_0_w - i)))
            else:
                if num_strings[i] == abs(n_0 + d * (n_0_w - i)) or num_strings[i] == num_strings[i - 1] - d \
                        or num_strings[i] == num_strings[i + 1] + d:
                    dealt_strings.append(num_strings[i])
                else:
                    dealt_strings.append(abs(n_0 + d * (n_0_w - i)))

    return dealt_strings


def find_most(list):
    most = 0
    count_list = []
    for i in range(len(list)):
        count_list.append(list.count(list[i]))
    most = list[count_list.index(max(count_list))]
    return most


class ScatterReader(AbstractGraphReader):
    def __init__(self, model_path='./best.pt'):
        self.yolo_model = YOLO(model_path)

    # 寻找图表原点及终点（左下及右上）
    # 传入为图像
    # 传出为原点坐标及终点坐标
    def find_boundary(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 3)
        edges = cv2.Canny(img, 30, 180, apertureSize=3, L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)
        h_lines = []  # 水平线
        p_lines = []  # 垂直线
        origin_point = []
        end_point = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1 == y2:
                h_lines.append(line)
            if x1 == x2:
                p_lines.append(line)

        # 找y轴
        x_min = 1000
        y_axis = []
        if len(p_lines):
            for line in p_lines:
                x1, y1, x2, y2 = line[0]
                if x1 < x_min:
                    x_min = x1
                    y_axis = [x1, y1, x2, y2]
            x1, y1, x2, y2 = y_axis
            origin_point.append(x1)
            end_point.append(y2)
            # print(f'y轴：x1:{x1},y1:{y1},x2:{x2},y2:{y2}')
            # cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            print('找不到垂直线')

        # 找x轴
        y_max = 10
        x_axis = []
        if len(h_lines):
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                if y1 > y_max:
                    y_max = y1
                    x_axis = [x1, y1, x2, y2]
            x1, y1, x2, y2 = x_axis
            origin_point.append(y1)
            end_point.append(x2)
            # print(f'x轴：x1:{x1},y1:{y1},x2:{x2},y2:{y2}')
            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            print('找不到水平线')

        origin_point = origin_point
        # cv2.circle(test_img1, (int(origin_point[0]), int(origin_point[1])), 2, (0, 0, 255), -1)
        end_point = end_point[::-1]
        # cv2.circle(test_img1, (int(end_point[0]), int(end_point[1])), 2, (0, 0, 255), -1)

        return origin_point, end_point

    # 找到图象中的散点，并传回其中心点的位置信息
    # 传入为图像
    # 传出应为一个数组,数组元素结构为[x位置，y位置]
    # '../../runs/detect/train11/weights/best.pt'
    def find_scatters(self, img, origin_point, end_point, yolo_model):
        scatters = []

        # yolo_model = YOLO(model_path)
        res = yolo_model(img)
        boxes = sorted(res[0].boxes, key=lambda b: b.xyxy[0][0].item())
        if len(boxes) > 0:
            for box in boxes:
                x_left_up = box.xyxy[0][0].item()
                y_left_up = box.xyxy[0][1].item()
                x_right_down = box.xyxy[0][2].item()
                y_right_down = box.xyxy[0][3].item()
                center_x = (x_left_up + x_right_down) / 2
                center_y = (y_left_up + y_right_down) / 2
                if origin_point[0] < center_x < end_point[0] and end_point[1] < center_y < origin_point[1]:
                    scatters.append([center_x, center_y])
        return scatters

    # 重写读取x，y轴坐标内容，x轴用easyocr读取，y轴用tesseract读取
    # 两者独立处理图像，独立处理数据，但返回内容均需为一维横向数组
    def read_x_axis(self, img, origin_point, end_point):
        label_result = []
        dealt_result = []
        num_strings = []

        n_0 = 0
        n_0_w = 0

        # 读取部分，easyocr需要cv2方式读入
        x_size, y_size = img.shape[1], img.shape[0]
        x_thresh = round(max(abs(origin_point[0]) / 2, abs(end_point[0] - x_size) / 2))
        y_thresh = round(abs(origin_point[1] - y_size) / 2)
        x_img = img[origin_point[1]:origin_point[1] + y_thresh, origin_point[0] - x_thresh:end_point[0] + x_thresh]
        x_img = cv2.resize(x_img, (0, 0), fx=2, fy=2)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)

        # easyocr读取数据
        # 可能存在的问题，',''.'混用，‘0’乱读等，存在多组数据读在一块
        # 解决办法：replace0.0，之后令其等差
        reader = easyocr.Reader(['en'])
        results = reader.readtext(x_img)

        # 处理多组数据沾连，替换部分字符
        for result in results:
            if (' ' in result[1]):
                temp = result[1].split()
                for a in temp:
                    dealt_result.append(a)
            else:
                temp = result[1].replace(',', '.')
                temp = temp.replace('O', '0')
                temp = temp.replace('o', '0')
                dealt_result.append(temp)

        for i in range(len(dealt_result)):
            try:
                n_0 = float(dealt_result[i])
                n_0_w = i
                num_strings.append(n_0)
            except Exception:
                num_strings.append(0.0)

        label_result = make_linear(num_strings, n_0, n_0_w, 0)
        return label_result

    def read_y_axis(self, img_path, origin_point, end_point):
        label_result = []
        dealt_result = []
        num_strings = []
        n_0 = 0
        n_0_w = 0

        image = Image.open(img_path)
        x_size, y_size = image.size[0], image.size[1]
        # x_thresh = round(max(abs(origin_point[0]) / 2, abs(end_point[0] - x_size) / 2))
        # y_thresh = round(abs(origin_point[1] - y_size) / 2)

        box1 = (max(origin_point[0] - 70, 0), end_point[1] - 10, origin_point[0], origin_point[1] + 10)
        pl_y_image = image.crop(box1)
        image = pl_y_image.resize((150, 500))

        config = r'-c tessedit_char_whitelist=0123456789. --psm 6'
        results = pytesseract.image_to_string(image, config=config)

        dealt_result = results.split()

        for i in range(len(dealt_result)):
            try:
                n_0 = float(dealt_result[i])
                n_0_w = i
                num_strings.append(n_0)
            except Exception:
                num_strings.append(0.0)

        label_result = make_linear(num_strings, n_0, n_0_w, 1)
        # label_result = results
        return label_result

    @staticmethod
    def get_id(filepath) -> str:
        # 获取文件名（包含扩展名）
        filename = os.path.basename(filepath)
        # 去除扩展名
        filename_without_ext = os.path.splitext(filename)[0]
        return filename_without_ext

    def read_graph(self, filepath) -> ReadResult:
        read_result: ReadResult = ReadResult()
        read_result.id = self.get_id(filepath)
        read_result.chart_type = GraphType.SCATTER.value
        try:
            img_path = filepath
            test_img = cv2.imread(filepath)
            scatters = []
            submit_x_series = []
            submit_y_series = []
            origin_point, end_point = self.find_boundary(test_img)
            scatters = self.find_scatters(test_img, origin_point, end_point, self.yolo_model)

            label_x_result = self.read_x_axis(test_img, origin_point, end_point)
            label_y_result = self.read_y_axis(img_path, origin_point, end_point)

            x_per_pixel = (label_x_result[-1] - label_x_result[0]) / (end_point[0] - origin_point[0])
            y_per_pixel = (label_y_result[0] - label_y_result[-1]) / (origin_point[1] - end_point[1])
            x0 = label_x_result[0]
            y0 = label_y_result[-1]

            for scatter in scatters:
                x = abs(scatter[0] - origin_point[0]) * x_per_pixel + x0
                y = abs(scatter[1] - origin_point[1]) * y_per_pixel + y0
                x_string = str(x)
                y_string = str(y)
                submit_x_series.append(x_string)
                submit_y_series.append(y_string)

            '''submit_x_series.replace(';', ',', 1)
            submit_x_series = submit_x_series + ',scatter'
            submit_y_series.replace(';', ',', 1)
            submit_y_series = submit_y_series + ',scatter'''''

            read_result.x_series = submit_x_series
            read_result.y_series = submit_y_series

            return read_result
        except LookupError as e:
            print(e)
            return ReadResult().default_result(filepath)


if __name__ == '__main__':
    #graph_reader = ScatterReader(model_path=ROOT_PATH + 'vertical_bar_reader/best.pt')
    # 0aa70ffb057f
    # -90 0e66aa993d9a
    #read_result = graph_reader.read_graph(DATASET_PATH + "train/images/00a5d76cb78d.jpg")
    graph_reader = ScatterReader()
    read_result = graph_reader.read_graph('../dataset/scatter/images'+'/00a5d76cb78d.jpg')
    print(read_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''test_image_path = '../dataset/scatter/images'
        img_name = '00a5d76cb78d.jpg'
        img_name2 = '00b423fdeea5.jpg'  # 非圆
        img_name3 = '00bc607d6b28.jpg'
        img_name4 = '00bde1b02364.jpg'
        img_name5 = '0fd1fdaf0390.jpg'  # 非圆'''
