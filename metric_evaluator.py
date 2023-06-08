import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz.distance.Levenshtein import distance as levenshtein
from sklearn.metrics import r2_score

import env
from abstract_graph_reader import AbstractGraphReader
from deplot_reader.deplot_reader import DeplotReader
from dot_reader.dot_reader import DotReader
from graph_classifier.graph_classifier_resnet import GraphClassifierResnet
from line_reader.line_reader import LineReader
from read_result import ReadResult
from vertical_bar_reader.vertical_bar_reader import VerticalBarReader
from graph_classifier.graph_classifier_lenet import GraphType, GraphClassifierLenet
from tqdm import tqdm


def sigmoid(x):
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true, y_pred):
    # The argument to the sigmoid transform is equal to
    # rmse(y_true, y_pred) / rmse(y_true, np.mean(y_true))
    return sigmoid((1 - r2_score(y_true, y_pred)) ** 0.5)


def normalized_levenshtein_score(y_true, y_pred):
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(y_true, y_pred):
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        return normalized_rmse(y_true, y_pred)


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame):
    """Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.

    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series`
        should be either arrays of floats or arrays of strings.

    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError("Must have exactly one prediction for each ground-truth instance.")
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(ground_truth.itertuples(index=False), predictions.itertuples(index=False))
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        if gt_type != pred_type:  # Check chart_type condition
            scores.append(0.0)
        else:  # Score with RMSE or Levenshtein as appropriate
            scores.append(score_series(gt_series, pred_series))
    return np.mean(scores)


def read_results_to_df(read_results, ground_truth):
    r = {}
    for read_result in read_results:
        read_result: ReadResult
        x_key, x_value, y_key, y_value = read_result.to_dict(ground_truth)
        r[x_key] = x_value
        r[y_key] = y_value
    return pd.DataFrame.from_dict(r, orient='index', columns=['data_series', 'chart_type']).rename_axis('id')


def ids_to_df(ids):
    r = {}
    for id in ids:
        x_key, x_value, y_key, y_value = annotation_to_dict(id)
        r[x_key] = x_value
        r[y_key] = y_value
    return pd.DataFrame.from_dict(r, orient='index', columns=['data_series', 'chart_type']).rename_axis('id')


def annotation_to_dict(id):
    with open(env.DATASET_PATH + f'train/annotations/{id}.json') as f:
        data = json.load(f)
        x_key = id + "_x"
        y_key = id + "_y"
        all_x = []
        all_y = []
        data_series = data["data-series"]
        chart_type = data['chart-type']
        if chart_type == GraphType.VERTICAL_BAR.value:
            # x为分类 y为数值
            for d in data_series:
                x = d['x']
                y = d['y']
                try:
                    y = float(y)
                except Exception as e:
                    print(e)
                    print(f"invalid y value in ground_truth, id = {id}")
                all_x.append(x)
                all_y.append(y)
        if chart_type == GraphType.HORIZONTAL_BAR.value:
            # x为数值 y为分类
            for d in data_series:
                x = d['x']
                y = d['y']
                try:
                    x = float(x)
                except Exception as e:
                    print(e)
                    print(f"invalid x value in ground_truth, id = {id}")
                all_x.append(x)
                all_y.append(y)
        if chart_type == GraphType.DOT.value:
            # The x-axis values will be numeric if the tick labels can be parsed as Python floats;
            # otherwise, they are categorical
            # y-axis 数值
            x_numeric_flag = True
            for d in data_series:
                x = d['x']
                try:
                    float(x)
                except:
                    x_numeric_flag = False
            for d in data_series:
                x = d['x']
                y = d['y']
                if x_numeric_flag:
                    all_x.append(float(x))
                else:
                    all_x.append(x)
                try:
                    y = float(y)
                except:
                    print(f"invalid y value in ground_truth, id = {id}")
                all_y.append(y)
        if chart_type == GraphType.LINE.value:
            # x分类 y数值
            for d in data_series:
                x = d['x']
                y = d['y']
                try:
                    y = float(y)
                except Exception as e:
                    print(e)
                    print(f"invalid y value in ground_truth, id = {id}")
                all_x.append(x)
                all_y.append(y)
        if chart_type == GraphType.SCATTER.value:
            # x数值 y数值
            for d in data_series:
                x = d['x']
                y = d['y']
                try:
                    x = float(x)
                except:
                    print(f"invalid x value in ground_truth, id = {id}")
                try:
                    y = float(y)
                except:
                    print(f"invalid y value in ground_truth, id = {id}")
                all_x.append(x)
                all_y.append(y)

        x_value = (all_x, chart_type)
        y_value = (all_y, chart_type)
    return x_key, x_value, y_key, y_value


def get_filenames_by_chart_types(filenames, chart_types):
    result_filenames = []
    for filename in tqdm(filenames, total=len(filenames), desc="Processing val datasets"):
        id, _ = os.path.splitext(filename)
        with open(env.DATASET_PATH + f'train/annotations/{id}.json') as f:
            data = json.load(f)
            chart_type = data['chart-type']
            if chart_type in chart_types:
                result_filenames.append(filename)
    return result_filenames


class MetricEvaluator():
    def __init__(self, val_ratio, chart_types=(GraphType.DOT.value,
                                               GraphType.LINE.value,
                                               GraphType.VERTICAL_BAR.value,
                                               GraphType.HORIZONTAL_BAR.value,
                                               GraphType.SCATTER.value)):
        self.val_ratio = val_ratio
        self.ground_truth: pd.DataFrame
        self.predictions: pd.DataFrame
        self.ids = []
        filenames = os.listdir(env.DATASET_PATH + "train/images")
        random.shuffle(filenames)
        filenames = get_filenames_by_chart_types(filenames, chart_types)
        filenames = filenames[:round(len(filenames) * val_ratio)]
        for filename in filenames:
            id, _ = os.path.splitext(filename)
            self.ids.append(id)
        self.ground_truth = ids_to_df(self.ids)

    def evaluate_reader(self, graph_reader):
        read_results = []
        total_iterations = len(self.ids)
        for id in tqdm(self.ids, total=total_iterations, desc='Processing images'):
            filepath = env.DATASET_PATH + f"train/images/{id}.jpg"
            try:
                read_result = graph_reader.read_graph(filepath)
            except Exception as e:
                print(e)
                print(f'error id = {id}')
                read_result = ReadResult().default_result(filepath)
            read_results.append(read_result)
        predictions = read_results_to_df(read_results, self.ground_truth)
        score = benetech_score(self.ground_truth, predictions)
        print(f'benetech_score = {score}')

    def evaluate_classifier(self, classifier):
        correct_cnt = 0
        for id in tqdm(self.ids, total=len(self.ids), desc='Processing images'):
            filepath = env.DATASET_PATH + f"train/images/{id}.jpg"
            graph_type = classifier.classify(filepath).value
            with open(env.DATASET_PATH + f'train/annotations/{id}.json') as f:
                data = json.load(f)
                chart_type = data['chart-type']
                if chart_type == graph_type:
                    correct_cnt += 1
        print(f'accuracy = {correct_cnt / len(self.ids)}')


if __name__ == '__main__':
    # reader = VerticalBarReader(env.ROOT_PATH + 'vertical_bar_reader/best.pt')
    # reader = DotReader()
    reader = LineReader()
    # reader = DeplotReader(GraphType.VERTICAL_BAR.value)
    # metric_evaluator = MetricEvaluator(0.001, GraphType.VERTICAL_BAR.value)
    # metric_evaluator = MetricEvaluator(0.05, GraphType.DOT.value)
    # metric_evaluator = MetricEvaluator(0.5, GraphType.HORIZONTAL_BAR.value)
    metric_evaluator = MetricEvaluator(0.1, GraphType.LINE.value)
    metric_evaluator.evaluate_reader(reader)

    '''
    graph_classifier = GraphClassifierResnet(env.ROOT_PATH + "graph_classifier/Benetech _ResNet50_fold0.pth")
    graph_classifier = GraphClassifierLenet(env.ROOT_PATH + "graph_classifier/graph_classifier.pth")
    metric_evaluator.evaluate_classifier(graph_classifier)
    '''

    '''
    ground_truth = pd.DataFrame.from_dict({
        '0a0a0_x': (['123.6', '456.7', '789.9'], 'dot'),
        '0a0a0_y': ([0.2, 0.9, 2.1], 'dot'),
        '0a0a1_x': (['123.6', '456.7', '789.9'], 'dot'),
        '0a0a1_y': ([0.2, 0.9, 2.1], 'dot'),
    }, orient='index', columns=['data_series', 'chart_type']).rename_axis('id')
    
    predictions = pd.DataFrame.from_dict({
        '0a0a0_x': (['123.6', '456.7', '789.9'], 'dot'),
        '0a0a0_y': ([0.0], 'dot'),
        '0a0a1_x': (['123.6', '456.7', '789.9'], 'dot'),
        '0a0a1_y': ([0.2, 0.9, 2.1], 'dot'),
    }, orient='index', columns=['data_series', 'chart_type']).rename_axis('id')
    benetech_score(ground_truth, predictions)
    '''
