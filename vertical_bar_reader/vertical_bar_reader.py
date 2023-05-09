import math
import os
from typing import List

import cv2
import easyocr
import numpy as np

from abstract_graph_reader import AbstractGraphReader
from graph_classifier.graph_classifier import GraphType
from read_result import ReadResult


class VerticalBarReader(AbstractGraphReader):
    def __init__(self):
        self.intersection = (0, 0)  # 坐标轴原点
        self.reader = easyocr.Reader(['en'])

    @staticmethod
    def is_horizontal_line(y1, y2):
        if y1 == y2:
            return True
        return False

    @staticmethod
    def get_x_axis_angle(img) -> int:
        detect_img = np.copy(img)
        # 转换为灰度
        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # 进行霍夫直线变换
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=15, maxLineGap=5)

        negative_45_cnt = 0
        positive_45_cnt = 0
        # 绘制直线
        for line in lines:
            x1, y1, x2, y2 = line[0]
            atan2 = math.atan2(y2 - y1, x2 - x1)
            angle = atan2 / math.pi * 180
            # print(f'angle = {angle}')
            if -50 < angle < -40:
                negative_45_cnt = negative_45_cnt + 1
            elif 40 < angle < 50:
                positive_45_cnt = positive_45_cnt + 1
            # cv2.line(detect_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 显示结果
        # cv2.imshow('detect_img', detect_img)
        if negative_45_cnt >= 2:
            return -45
        if positive_45_cnt >= 2:
            return 45
        return 0

    def read_x_axis(self, x_axis_img) -> List[str]:
        angle = self.get_x_axis_angle(x_axis_img)
        rows, cols = x_axis_img.shape[:2]
        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

        # 计算输出图像的大小
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_cols = int(rows * sin + cols * cos)
        new_rows = int(rows * cos + cols * sin)
        M[0, 2] += (new_cols - cols) / 2
        M[1, 2] += (new_rows - rows) / 2

        # 执行仿射变换
        rotated_img = cv2.warpAffine(x_axis_img, M, (new_cols, new_rows))
        resize_x_axis_img = cv2.resize(rotated_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # cv2.namedWindow('x_axis_img')
        # cv2.imshow('x_axis_img', resize_x_axis_img)
        # 读取图片并进行识别
        ocr_result = self.reader.readtext(resize_x_axis_img)
        ocr_result = sorted(ocr_result, key=lambda c: c[0][0][1])
        x_axis_result = []
        # 输出识别结果
        for r in ocr_result:
            # r[0]表示文本行的四个顶点坐标，按照左上、右上、右下、左下的顺序排列。
            # r[1]表示字符串
            # r[2]表示置信度
            if abs(r[0][0][1] - r[0][1][1]) < 5:  # 只取水平线方向的文本
                x_axis_result.append(r[1])
        return x_axis_result

    def get_value_per_pixel(self, y_axis_img) -> int:
        # 先放大，若不放大easyOCR的识别度较低
        resize_factor = 2
        resized = cv2.resize(y_axis_img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        # cv2.namedWindow('get_value_per_pixel')
        # cv2.imshow('get_value_per_pixel', resized)
        ocr_result = self.reader.readtext(resized)
        reference_math = None
        for r in ocr_result:
            if r[1].isdigit() and r[2] >= 0.8 and r[0][0][1] < self.intersection[1]:
                reference_math = r
                break
        if reference_math is None:
            raise LookupError("未定位到数字")
        reference_math_pos = reference_math[0]
        reference_math_value = reference_math[1]
        value_per_pixel = int(reference_math_value) / (self.intersection[1] - reference_math_pos[0][1])
        return value_per_pixel / resize_factor

    def read_bar(self, img, value_per_pixel) -> List[str]:
        img_height, img_width, _ = img.shape
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 阈值处理
        thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # 轮廓检测
        contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        y_axis_result: List[str] = []
        # 遍历轮廓
        for contour in contours:
            # 计算轮廓的矩形边界框
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)  # box是左下 左上 右上 右下的坐标顺序
            # 计算矩形宽度和高度
            width = rect[1][0]
            height = rect[1][1]
            # 进行矩形检查
            if width * height < img_width * img_height * 0.1:
                v = (self.intersection[1] - box[1][1]) * value_per_pixel
                y_axis_result.append(str(v))
                # print(box)
                # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        return y_axis_result

    def get_id(self, filepath) -> str:
        # 获取文件名（包含扩展名）
        filename = os.path.basename(filepath)
        # 去除扩展名
        filename_without_ext = os.path.splitext(filename)[0]
        return filename_without_ext

    def read_graph(self, filepath) -> ReadResult:
        # 读取图片
        img = cv2.imread(filepath)
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # 直线检测
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=3)

        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if self.is_horizontal_line(y1, y2):
                horizontal_lines.append((x1, y1, x2, y2))
        horizontal_axis = max(horizontal_lines, key=lambda l: l[1])
        intersection = (horizontal_axis[0], horizontal_axis[1])
        self.intersection = intersection
        height, width, channels = img.shape
        y_axis_img = img[0:height, 0:intersection[0]]
        value_per_pixel = self.get_value_per_pixel(y_axis_img)
        y_axis_result = self.read_bar(img, value_per_pixel)
        x_axis_img = img[intersection[1]:height, 0:width]
        x_axis_result = self.read_x_axis(x_axis_img)
        read_result: ReadResult = ReadResult()
        read_result.id = self.get_id(filepath)
        read_result.x_series = x_axis_result
        read_result.y_series = y_axis_result
        read_result.chart_type = GraphType.VERTICAL_BAR.value
        return read_result


if __name__ == '__main__':
    graph_reader = VerticalBarReader()
    read_result = graph_reader.read_graph("dataset/train/images/000e8130e62a.jpg")
    print(read_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
