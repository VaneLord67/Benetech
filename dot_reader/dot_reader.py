import re
import math
import os
from typing import List

import cv2
import easyocr
import numpy as np

from env import *
from abstract_graph_reader import AbstractGraphReader
from graph_classifier.graph_classifier_lenet import GraphType
from read_result import ReadResult, is_value


# 根据路径加载图片并求其二值图像
def load_image(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU表示将处理结果反转并自适应阈值处理
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if DEBUG_MODE:
        cv2.imshow('img_binary', binary)
    return image, binary


# 形态学变换，传入参数为形态学操作函数所需的参数，以字典类型传入
def morph_transform(params):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params['kernel_size'])
    output = cv2.morphologyEx(params['input'], params['operation'], kernel, iterations=params['iterations'])
    return output


# 边缘检测，返回边缘的像素点集
def contour_detection(params):
    contours = cv2.findContours(params['input'], mode=params['mode'], method=params['approx'])
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


# 条件过滤，滤除不符合条件的边缘
def boolean_filter(data, condition):
    idx = np.where(condition)[0]
    return [d for i, d in enumerate(data) if i in idx]


# 检测填充物的边缘，通过设置形态学开运算的kernel_size，对坐标轴进行滤除
def filled_shapes_detection(img):
    morph_params = {'kernel_size': (3, 3), 'input': img, 'operation': cv2.MORPH_OPEN, 'iterations': 2}
    contour_params = {'input': morph_transform(morph_params), 'mode': cv2.RETR_TREE, 'approx': cv2.CHAIN_APPROX_NONE}
    contours = contour_detection(contour_params)
    return contours


# 坐标轴检测，通过设置形态学开运算的kernel_size，对横边缘或者竖边缘进行滤除
def axis_detection(img, min_thresh, max_thresh, axis='x'):
    kernel_size = (9, 1) if axis == 'x' else (1, 9)
    ind = 0 if axis == 'x' else 1
    morph_params = {'kernel_size': kernel_size, 'input': img, 'operation': cv2.MORPH_OPEN, 'iterations': 2}
    contour_params = {'input': morph_transform(morph_params), 'mode': cv2.RETR_TREE, 'approx': cv2.CHAIN_APPROX_NONE}
    contours = contour_detection(contour_params)
    if not contours:
        return []
    length = np.array([c[:, :, ind].max() - c[:, :, ind].min() for c in contours])
    # 先用长度范围进行过滤
    contours = boolean_filter(contours, (min_thresh < length) & (length < max_thresh))
    if not contours:
        raise LookupError("未检测到x轴")
    # 再用方位进行过滤，即x轴应该在最下方，y轴应该在最左方
    if len(contours) > 1:
        if ind:  # 如果是要检测y轴，则取x值最小的轮廓
            far_points = np.array([c[:, :, 0].min() for c in contours])
            contours = boolean_filter(contours, far_points == min(far_points))
        else:  # 如果是要检测x轴，则取y值最大的轮廓
            far_points = np.array([c[:, :, 1].max() for c in contours])
            contours = boolean_filter(contours, far_points == max(far_points))
    return contours


def get_intersection(x_axis_contour):
    if len(x_axis_contour) > 1:
        return 0, 0
    intersection_x = x_axis_contour[0][:, :, 0].min()
    intersection_y = round(x_axis_contour[0][:, :, 1].mean())
    if DEBUG_MODE:
        print(f'intersection_x={intersection_x},intersection_y={intersection_y}')
    return intersection_x, intersection_y


def data_submit_check(y_series):
    submit_y_series = [str(num) for num in y_series]
    return submit_y_series


def data_align(x_axis_result_pos_x, x_axis_result, y_map, intersection):
    align_y_axis_result = []
    align_x_axis_result = []
    dot_pos_x = y_map['x_center']
    dot_count = y_map['y_count']
    for idx, pos_x in enumerate(x_axis_result_pos_x):
        if pos_x < intersection[0]:
            continue
        flag = False
        align_x_axis_result.append(x_axis_result[idx])
        for i in range(len(dot_pos_x)):
            dot_x_center = dot_pos_x[i]
            if abs(dot_x_center - pos_x) < 5:  # 避免在有x的坐标值，但该x坐标值下面无dot点的情况下写乱
                align_y_axis_result.append(dot_count[i])
                flag = True
                break
        if not flag:
            align_y_axis_result.append('0')
    return align_x_axis_result, align_y_axis_result


def get_x_axis_img(img, intersection):
    height, width, _ = img.shape
    return img[intersection[1]:height, 0:width]


def get_y_axis_img(img, intersection):
    height, width, _ = img.shape
    return img[0:height, 0:intersection[0]]


def rotate_image(img, angle):
    height, width = img.shape[:2]
    rotation_angle_radian = np.deg2rad(angle)

    new_width = int(width * np.abs(np.cos(rotation_angle_radian)) + height * np.abs(np.sin(rotation_angle_radian)))
    new_height = int(width * np.abs(np.sin(rotation_angle_radian)) + height * np.abs(np.cos(rotation_angle_radian)))

    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))
    return rotated_img, rotation_matrix


def rotate_point(point, rotation_matrix):
    point = np.array([[point[0]], [point[1]], [1]])
    transformed_point = np.dot(rotation_matrix, point)
    return int(transformed_point[0]), int(transformed_point[1])


# 过滤候选边缘，滤除小的噪声边缘(threshold = 0.8 * max(area))
def filter_dots(candidates):
    if not len(candidates):
        return []
    area = np.array([cv2.contourArea(c) for c in candidates])
    threshold = 0.8 * max(area)
    return boolean_filter(candidates, area > threshold)


# 数每个x值对应几个dot，返回一个字典类型
def count_dots(dot_contours):
    if not len(dot_contours):
        raise LookupError("未检测到dot边缘")
    dots = [cv2.minEnclosingCircle(c) for c in dot_contours]
    centers = np.array([np.int32(np.round(d[0])) for d in dots])
    radii = np.array([np.int32(np.round(d[1])) for d in dots])
    tolerance = int(np.mean(radii))  # 以平均半径长为容忍度
    center_x = centers[:, 0]  # 第二维是0表示x轴坐标，是1表示y轴坐标
    mapping = {'x_center': [], 'y_count': []}
    move = True
    while move:
        min_x = min(center_x)
        idx = np.where(np.abs(center_x - min_x) < tolerance)[0]
        mapping['x_center'].append(min_x)
        mapping['y_count'].append(len(idx))
        center_x = [c for i, c in enumerate(center_x) if i not in idx]
        move = True if center_x else False
    return mapping


# 在所有的dot边缘点中，找到最大的x值和最小的x值，两值相减作为x轴长度的最小阈值
def get_min_threshold(dot_contours):
    end = max([c[:, :, 0].max() for c in dot_contours])
    start = min([c[:, :, 0].min() for c in dot_contours])
    return end - start


# 由于识别出来的文字可能存在两个或多个x坐标文字合体的情况，也可能存在一个x坐标文字有两行但却分开框起来的情况，故进行一些处理
def process_ocr_result(ocr_result):
    # 先进行合体文字拆分
    ocr_end_idx = len(ocr_result) - 1
    for idx, r in enumerate(ocr_result):
        if '  ' in r[1]:
            long_text = r[1]
            if '   ' in long_text:
                if '    ' in long_text:
                    texts=long_text.split('    ')
                else:
                    texts = long_text.split('   ')
            else:
                texts = long_text.split('  ')
            texts_len = len(texts)
            # 增加list长度，避免超过最大索引值而赋不了值
            for i in range(texts_len - 1):
                ocr_result.append(tuple([1, 2, 3]))
            ocr_result[idx + texts_len:ocr_end_idx + texts_len] = ocr_result[idx + 1:ocr_end_idx + 1]
            i = 0
            while i < texts_len:
                # 由于不能对元组进行赋值操作，故先转化为列表，处理完之后再转化为元组
                list_ocr_result = list(ocr_result[idx + i])
                list_ocr_result[1] = texts[i]
                list_ocr_result[2] = ocr_result[idx][2]
                # 将合体的文字平均分块
                mean_text_len = round((r[0][1][0] - r[0][0][0]) / texts_len)
                list_ocr_result[0] = (
                [ocr_result[idx + i - 1 if i else idx][0][1 if i else 0][0], ocr_result[idx][0][0][1]],
                [ocr_result[idx + i - 1 if i else idx][0][1 if i else 0][0] + mean_text_len, ocr_result[idx][0][0][1]],
                [ocr_result[idx + i - 1 if i else idx][0][1 if i else 0][0] + mean_text_len, ocr_result[idx][0][3][1]],
                [ocr_result[idx + i - 1 if i else idx][0][1 if i else 0][0], ocr_result[idx][0][3][1]])
                ocr_result[idx + i] = tuple(list_ocr_result)
                i += 1

    # 再进行两行的未合体的文字的合成
    # 首先判断有几行文字，只有超过两行的要进行文字合成处理
    level_y = []
    for idx, r in enumerate(ocr_result):
        if len(level_y) == 0:
            level_y.append(r[0][0][1])
        y_difference = min([abs(r[0][0][1] - level_value) for level_value in level_y])
        if y_difference > 5:
            level_y.append(r[0][0][1])
        else:
            continue
    level_y = sorted(level_y)
    level_num = len(level_y)
    # 当有超过两行文字时，找到中间那一行每个文字框的正上方的框，把文字值嵌入进去，并删除中间这一行的框
    idx_list = []
    if level_num > 2:
        for idx, r in enumerate(ocr_result):
            if abs(r[0][0][1]-level_y[1]) < 5:
                idx_list.append(idx)
                for idx1, r1 in enumerate(ocr_result):
                    if r1[0][1][0] > (r[0][0][0] + r[0][1][0])/2 > r1[0][0][0] and abs(r1[0][0][1] - level_y[0]) < 5:
                        list_ocr_result = list(r1)
                        list_ocr_result[1] = list_ocr_result[1] + ' ' + r[1]
                        ocr_result[idx1] = tuple(list_ocr_result)
                        break
        deleted_ocr_result = []
        for i in range(len(ocr_result)):
            if i not in idx_list:
                deleted_ocr_result.append(ocr_result[i])
        ocr_result = deleted_ocr_result
    return ocr_result


def morphology_method(x_axis_img) -> int:
    # 读取图像
    resize_factor = 3
    x_axis_img = cv2.resize(x_axis_img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    img_height, img_width, _ = x_axis_img.shape
    # 转换为灰度图像
    gray = cv2.cvtColor(x_axis_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 创建CLAHE对象
    gray = clahe.apply(gray)  # 自适应直方图均衡化
    # 阈值处理
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
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
        if -55 < angle < -35:
            negative_45_cnt += 1
        elif 35 < angle < 55:
            positive_45_cnt += 1
        elif -100 < angle < -80:
            negative_90_cnt += 1
        elif -10 < angle < 10:
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
    elif negative_90_cnt / zero_cnt >= 1:
        return -90
    else:
        return 0


def get_x_axis_angle(img) -> int:
    return morphology_method(img)


class DotReader(AbstractGraphReader):
    def __init__(self):
        self.intersection = (0, 0)  # 坐标轴原点
        self.reader = READER

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

    def read_x_axis(self, x_axis_img) -> (List[str], List[int]):
        angle = get_x_axis_angle(x_axis_img)
        if DEBUG_MODE:
            print(f'x_axis_angle = {angle}')
        rotated_img, rotation_matrix = rotate_image(x_axis_img, angle)  # 如果不执行旋转变换，那么ocr可能会识别到断续的结果
        resize_factor = 2
        resize_x_axis_img = cv2.resize(rotated_img, None,
                                       fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
        if DEBUG_MODE:
            cv2.imshow('x_axis_img', resize_x_axis_img)
        # 读取图片并进行识别
        ocr_result = self.reader.readtext(resize_x_axis_img)
        ocr_result = [r for r in ocr_result if r[2] >= 0.1]  # 置信度约束
        if angle == -45:
            ocr_result = sorted(ocr_result, key=lambda c: c[0][0][1])
        elif angle == 45:
            ocr_result = sorted(ocr_result, key=lambda c: -c[0][0][1])
        elif angle == 0:
            ocr_result = sorted(ocr_result, key=lambda c: c[0][0][0])
        elif angle == -90:
            ocr_result = sorted(ocr_result, key=lambda c: c[0][0][1])
        # 由于识别出来的文字可能存在两个或多个x坐标文字合体的情况，也可能存在一个x坐标文字有两行但却分开框起来的情况，故进行一些处理
        ocr_result = process_ocr_result(ocr_result)

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
        x_axis_result_pos_x = []
        rotation_matrix_3x3 = np.vstack([rotation_matrix, [0, 0, 1]])  # 添加齐次坐标的最后一行
        rotation_matrix_inv = np.linalg.inv(rotation_matrix_3x3)
        if angle == 0 and len(ocr_result) == 1:
            long_text = ocr_result[0][1]
            texts = long_text.split(' ')
            return [t for t in texts], None
        for r in ocr_result:
            # r[0]表示文本行的四个顶点坐标，按照左上、右上、右下、左下的顺序排列。
            # r[1]表示字符串
            # r[2]表示置信度
            if abs(r[0][0][1] - r[0][1][1]) < 5:  # 只取水平线方向的文本
                if angle == 0 and r[0][0][1] > aver_ocr_result_y:
                    if DEBUG_MODE:
                        print(f'skip text: {r[1]}')
                    continue
                if angle == -45:
                    result_pos_x = r[0][1][0]
                    result_pos_y = r[0][1][1]
                elif angle == 45:
                    result_pos_x = r[0][0][0]
                    result_pos_y = r[0][0][1]
                elif angle == 0:
                    result_pos_x = (r[0][0][0] + r[0][1][0]) / 2
                    result_pos_y = (r[0][0][1] + r[0][1][1]) / 2
                elif angle == -90:
                    result_pos_x = (r[0][0][0] + r[0][1][0]) / 2
                    result_pos_y = (r[0][0][1] + r[0][3][1]) / 2
                else:
                    raise LookupError("unsupported angle")
                original_x, original_y = rotate_point((result_pos_x / resize_factor, result_pos_y / resize_factor),
                                                      rotation_matrix_inv)
                x_axis_result_pos_x.append(original_x)
                x_axis_result.append(r[1])
        if DEBUG_MODE:
            print(f'x_axis_result_pos_x = {x_axis_result_pos_x}')
        return x_axis_result, x_axis_result_pos_x

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
        read_result.chart_type = GraphType.DOT.value
        try:
            # 读取图片及其二值图像
            img, img_binary = load_image(filepath)
            # 获取候选边缘
            candidates = filled_shapes_detection(img_binary)
            # 对候选边缘进行过滤，滤除小的杂点
            dot_contours = filter_dots(candidates)
            if DEBUG_MODE:
                cv2.drawContours(image=img, contours=dot_contours, contourIdx=-1, color=(0, 0, 255),
                                 thickness=2, lineType=cv2.LINE_AA)
                cv2.imshow('dot_contours', img)
            # 得到每个x中心对应的点的个数(输出结果是字典形式)
            y_map = count_dots(dot_contours)
            print(y_map)

            # 接下来检测x轴位置
            if dot_contours:  # 获取x轴长度的最小阈值
                min_threshold = get_min_threshold(dot_contours)
            else:
                min_threshold = int(img_binary.shape[1]) / 4
            max_threshold = int(img_binary.shape[1]) - 2  # 获取x轴长度的最大阈值
            x_axis_contour = axis_detection(img_binary, min_threshold, max_threshold, axis='x')
            if DEBUG_MODE:
                cv2.drawContours(image=img, contours=x_axis_contour, contourIdx=-1, color=(0, 0, 255),
                                 thickness=2, lineType=cv2.LINE_AA)
                cv2.imshow('x_axis_detect', img)
            intersection_x, intersection_y = get_intersection(x_axis_contour)
            intersection = (intersection_x, intersection_y)
            # 接下来读取x轴的坐标文字
            x_axis_img = get_x_axis_img(img, intersection)
            x_axis_result, x_axis_result_pos_x = self.read_x_axis(x_axis_img)
            if x_axis_result_pos_x:
                x_axis_result, y_axis_result = data_align(x_axis_result_pos_x,
                                                          x_axis_result, y_map, intersection)
            else:
                y_axis_result = y_map['y_count']
            read_result.x_series, read_result.y_series = x_axis_result, data_submit_check(y_axis_result)
            return read_result
        except LookupError as e:
            print(e)
            return ReadResult().default_result(filepath)


if __name__ == '__main__':
    graph_reader = DotReader()

    # 9c1230c5200e  17f86b2b22bb文字有两行
    # 08e601f45bf3 9f73d35cdcf0 34fdf448d85b最后两个文字最后无法分开
    read_result = graph_reader.read_graph(DATASET_PATH + "train/dot/images/0a2dd5cbd239.jpg")
    print(read_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
