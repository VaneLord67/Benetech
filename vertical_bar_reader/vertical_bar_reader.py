import math
import os
from typing import List

import cv2
import easyocr
import numpy as np

from abstract_graph_reader import AbstractGraphReader
from graph_classifier.graph_classifier import GraphType
from read_result import ReadResult

DEBUG_MODE = True


def on_mouse(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONUP:
        print("鼠标左键单击：(%d, %d)" % (x, y))


def is_horizontal_line(y1, y2):
    if y1 == y2:
        return True
    return False


def is_vertical_line(x1, x2):
    if x1 == x2:
        return True
    return False


def get_x_axis_img(img, intersection):
    height, width, _ = img.shape
    return img[intersection[1]:height, 0:width]


def get_y_axis_img(img, intersection):
    height, width, _ = img.shape
    return img[0:height, 0:intersection[0]]


def background_is_grey_or_black(img) -> bool:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean = cv2.mean(gray_img)[0]
    threshold = 150  # 阈值可以根据具体情况进行调整
    if gray_mean < threshold:
        if DEBUG_MODE:
            print('Image background is dark or gray')
        return True
    else:
        if DEBUG_MODE:
            print('Image background is light or bright')
        return False


def invert_img(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lut = np.zeros((256, 1), dtype=np.uint8)
    for i in range(256):
        lut[i][0] = 255 - i
    inverted_img = cv2.LUT(gray_img, lut)
    inverted_img = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)
    cv2.imshow('invert_img', inverted_img)
    return inverted_img


def get_intersection(img):
    intersection_img = np.copy(img)
    intersection_img_height, intersection_img_width, _ = intersection_img.shape
    # 转换为灰度图像
    gray = cv2.cvtColor(intersection_img, cv2.COLOR_BGR2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 直线检测
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=3)

    horizontal_lines = []
    vertical_lines = []
    # 在原始图像上绘制直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if is_horizontal_line(y1, y2) and abs(intersection_img_height - y1) > 5:
            cv2.line(intersection_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            horizontal_lines.append((x1, y1, x2, y2))
        if is_vertical_line(x1, x2) and abs(intersection_img_width - x1) > 5:
            cv2.line(intersection_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            vertical_lines.append((x1, y1, x2, y2))
    if DEBUG_MODE:
        cv2.imshow('intersection_img', intersection_img)
    if len(horizontal_lines) > 0:
        horizontal_axis = max(horizontal_lines, key=lambda l: (abs(x2 - x1), l[1]))
        intersection = (horizontal_axis[0], horizontal_axis[1])
        return intersection
    elif len(vertical_lines) > 0:
        vertical_axis = max(vertical_lines, key=lambda l: (abs(y2 - y1), l[0]))
        intersection = (vertical_axis[0], vertical_axis[1])
        return intersection
    raise LookupError("未定位到坐标轴原点")


def rotate_x_axis_img(x_axis_img, angle):
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
    return rotated_img


def morphology_method(x_axis_img) -> int:
    # 读取图像
    should_invert = background_is_grey_or_black(x_axis_img)
    if should_invert:
        x_axis_img = invert_img(x_axis_img)
    resize_factor = 3
    x_axis_img = cv2.resize(x_axis_img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    img_height, img_width, _ = x_axis_img.shape
    # 转换为灰度图像
    gray = cv2.cvtColor(x_axis_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 创建CLAHE对象
    gray = clahe.apply(gray)  # 自适应直方图均衡化
    # 阈值处理
    thresh = cv2.threshold(gray, 80 if should_invert else 150, 255, cv2.THRESH_BINARY)[1]
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # 轮廓检测
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0
    area_cnt = 0
    for contour in contours:
        # 计算轮廓的矩形边界框
        rect = cv2.minAreaRect(contour)
        # 计算矩形宽度和高度
        width = rect[1][0]
        height = rect[1][1]
        # 进行矩形检查
        area = width * height
        if area > img_width * img_height * 0.5:
            continue
        area_cnt += 1
        total_area += area
    aver_area = total_area / (area_cnt if area_cnt > 0 else 1)

    negative_45_cnt = 0
    positive_45_cnt = 0
    negative_90_cnt = 0
    zero_cnt = 0
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
        area = width * height
        if area > img_width * img_height * 0.5 or area < 10 or area < aver_area:
            continue
        distance1 = cv2.norm(np.array(box[0]), np.array(box[1]), cv2.NORM_L2)
        distance2 = cv2.norm(np.array(box[1]), np.array(box[2]), cv2.NORM_L2)
        if distance1 > distance2:
            x1, y1 = box[0]
            x2, y2 = box[1]
        else:
            x1, y1 = box[1]
            x2, y2 = box[2]
        atan2 = math.atan2(y2 - y1, x2 - x1)
        angle = atan2 / math.pi * 180
        if height > width and -5 < angle < 5:
            angle = -90
        if -50 < angle < -40:
            negative_45_cnt += 1
        elif 40 < angle < 50:
            positive_45_cnt += 1
        elif -95 < angle < -85:
            negative_90_cnt += 1
        elif -5 < angle < 5:
            zero_cnt += 1
        if DEBUG_MODE:
            cv2.drawContours(x_axis_img, [box], 0, (0, 255, 0), 2)
    zero_cnt = max(zero_cnt, 1)
    if DEBUG_MODE:
        cv2.imshow('get_x_angle_img', x_axis_img)
    if negative_45_cnt / zero_cnt >= 0.5:
        return -45
    elif positive_45_cnt / zero_cnt >= 0.5:
        return +45
    elif negative_90_cnt / zero_cnt >= 0.5:
        return -90
    else:
        return 0


def get_x_axis_angle(img) -> int:
    return morphology_method(img)


def is_value(value_string) -> bool:
    s = value_string.split('.')
    if len(s) > 2:
        return False
    else:
        for si in s:
            if not si.isdigit():
                return False
        return True


def filter_bar(rect, box, img, intersection):
    img_height, img_width, _ = img.shape
    # 计算矩形宽度和高度
    width = rect[1][0]
    height = rect[1][1]
    # 进行矩形检查
    area = width * height
    img_area = img_width * img_height
    # 剔除小的噪声矩形，剔除最外围的包括整个图像的矩形
    if 30 <= area < img_area * 0.1 and box[0][0] > intersection[0] and box[0][1] <= intersection[1]:
        return True
    return False


def is_bar_domain(l, intersection) -> bool:
    x1, y1, x2, y2 = l
    y = max(y1, y2)
    x = min(x1, x2)
    if x > intersection[0] and abs(y - intersection[1]) < 5:
        return True
    return False


def split_bar_line_method(img, intersection):
    split_bar_line_img = np.copy(img)
    # 转换为灰度图像
    gray = cv2.cvtColor(split_bar_line_img, cv2.COLOR_BGR2GRAY)
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 直线检测
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=3, maxLineGap=3)
    lines = sorted(lines, key=lambda k: k[0][0])  # 按x排列
    bars = []
    dot_group = []
    # 在原始图像上绘制直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y1 <= y2:
            dot = (x1, y1)
        else:
            dot = (x2, y2)
        if is_vertical_line(x1, x2) and is_bar_domain((line[0]), intersection):
            dot_group.append(dot)
            if len(dot_group) == 2:
                bars.append((dot_group[0], dot_group[1]))
                dot_group = []
            cv2.line(split_bar_line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if DEBUG_MODE:
        cv2.imshow("split_bar_line_img", split_bar_line_img)
        cv2.setMouseCallback('split_bar_line_img', on_mouse)
        print(f'bars = {bars}')
    return bars


def split_bar_contour_method(img, intersection):
    split_bar_contour_method_img = np.copy(img)
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(8, 8))  # 创建CLAHE对象
    gray = clahe.apply(gray)
    # 阈值处理
    thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # 轮廓检测
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    bars = []
    for contour in contours:
        # 计算轮廓的矩形边界框
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # box是左下 左上 右上 右下的坐标顺序
        if filter_bar(rect, box, img, intersection):
            bars.append((box[1], box[2]))
            if DEBUG_MODE:
                cv2.drawContours(split_bar_contour_method_img, [box], 0, (0, 255, 0), 2)
    if DEBUG_MODE:
        cv2.imshow('split_bar_contour_method_img', split_bar_contour_method_img)
    return bars


class VerticalBarReader(AbstractGraphReader):
    def __init__(self):
        self.intersection = (0, 0)  # 坐标轴原点
        self.reader = easyocr.Reader(['en'])

    @staticmethod
    def hough_method(img) -> int:
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
            if -50 < angle < -40:
                negative_45_cnt = negative_45_cnt + 1
            elif 40 < angle < 50:
                positive_45_cnt = positive_45_cnt + 1
            # cv2.line(detect_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 显示结果
        if DEBUG_MODE:
            cv2.imshow('detect_img', detect_img)
        if negative_45_cnt >= 2:
            return -45
        if positive_45_cnt >= 2:
            return 45
        return 0

    def read_x_axis(self, x_axis_img) -> List[str]:
        angle = get_x_axis_angle(x_axis_img)
        if DEBUG_MODE:
            print(f'angle = {angle}')
        rotated_img = rotate_x_axis_img(x_axis_img, angle)  # 如果不执行旋转变换，那么ocr可能会识别到断续的结果
        resize_factor = 2
        resize_x_axis_img = cv2.resize(rotated_img, None,
                                       fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        if DEBUG_MODE:
            cv2.imshow('x_axis_img', resize_x_axis_img)
        # 读取图片并进行识别
        ocr_result = self.reader.readtext(resize_x_axis_img)
        if angle == -45:
            ocr_result = sorted(ocr_result, key=lambda c: c[0][0][1])
        elif angle == 45:
            ocr_result = sorted(ocr_result, key=lambda c: -c[0][0][1])
        elif angle == 0:
            ocr_result = sorted(ocr_result, key=lambda c: c[0][0][0])
        else:
            ocr_result = sorted(ocr_result, key=lambda c: c[0][0][0])
        x_axis_result = []
        aver_ocr_result_y = 0
        if angle == 0:
            ocr_result_y_cnt = 0
            ocr_result_y_sum = 0
            for r in ocr_result:
                ocr_result_y_sum += r[0][0][1]
                ocr_result_y_cnt += 1
            aver_ocr_result_y = ocr_result_y_sum / ocr_result_y_cnt if ocr_result_y_cnt > 0 else 1
        if DEBUG_MODE:
            print(f'ocr_result = {ocr_result}')
        # 输出识别结果
        for r in ocr_result:
            # r[0]表示文本行的四个顶点坐标，按照左上、右上、右下、左下的顺序排列。
            # r[1]表示字符串
            # r[2]表示置信度
            if abs(r[0][0][1] - r[0][1][1]) < 5 and r[2] >= 0.3:  # 只取水平线方向的文本
                if angle == 0 and r[0][0][1] > aver_ocr_result_y:
                    continue
                x_axis_result.append(r[1])
        return x_axis_result

    def get_value_per_pixel(self, y_axis_img) -> int:
        # 先放大，若不放大easyOCR的识别度较低
        resize_factor = 2
        resized = cv2.resize(y_axis_img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        ocr_result = self.reader.readtext(resized)
        reference_math = None
        for r in ocr_result:
            if is_value(r[1]) and r[2] >= 0.8 and r[0][0][1] < self.intersection[1]:
                reference_math = r
                break
        if reference_math is None:
            raise LookupError("未定位到数字")
        reference_math_pos = reference_math[0]
        reference_math_pos = [[x / resize_factor for x in sub_list] for sub_list in reference_math_pos]
        reference_math_value = reference_math[1]
        value_per_pixel = float(reference_math_value) / (self.intersection[1] - reference_math_pos[0][1])
        if DEBUG_MODE:
            cv2.imshow('get_value_per_pixel', resized)
            print(f'定位到数字: {reference_math_value}, 位置: {reference_math_pos}')
            print(f'value_per_pixel = {value_per_pixel}')
        return value_per_pixel

    def read_bar(self, img, value_per_pixel) -> List[str]:
        bars = split_bar_line_method(img, self.intersection)
        y_axis_result: List[str] = []
        # 遍历轮廓
        for bar in bars:
            v = (self.intersection[1] - min(bar[1][1], bar[0][1])) * value_per_pixel
            y_axis_result.append(str(v))
            # print(box)
        if DEBUG_MODE:
            cv2.imshow("read_bar", img)
        return y_axis_result

    @staticmethod
    def get_id(filepath) -> str:
        # 获取文件名（包含扩展名）
        filename = os.path.basename(filepath)
        # 去除扩展名
        filename_without_ext = os.path.splitext(filename)[0]
        return filename_without_ext

    def read_graph(self, filepath) -> ReadResult:
        # 读取图片
        img = cv2.imread(filepath)
        intersection = get_intersection(img)
        self.intersection = intersection
        y_axis_img = get_y_axis_img(img, intersection)
        value_per_pixel = self.get_value_per_pixel(y_axis_img)
        y_axis_result = self.read_bar(img, value_per_pixel)
        x_axis_img = get_x_axis_img(img, intersection)
        x_axis_result = self.read_x_axis(x_axis_img)
        read_result: ReadResult = ReadResult()
        read_result.id = self.get_id(filepath)
        # 这里让x轴和y轴读取的数据进行对齐
        # min_len = min(len(x_axis_result), len(y_axis_result))
        read_result.x_series = x_axis_result
        read_result.y_series = y_axis_result
        read_result.chart_type = GraphType.VERTICAL_BAR.value
        return read_result


if __name__ == '__main__':
    DEBUG_MODE = True
    graph_reader = VerticalBarReader()
    try:
        # 0aa70ffb057f
        read_result = graph_reader.read_graph("dataset/train/images/000e8130e62a.jpg")
        print(read_result)
    except LookupError as e:
        print(e)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
