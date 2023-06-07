import os
from typing import List

from graph_classifier.graph_classifier_lenet import GraphType


def is_value(value_string) -> bool:
    try:
        float(value_string)
        return True
    except:
        return False


class ReadResult:
    def __init__(self):
        # 注意，ReadResult中处理的都是字符串，如果某些坐标数据是数字浮点类型，要先转成字符串。
        self.id: str = ""
        self.x_series: List[str] = []
        self.y_series: List[str] = []
        self.chart_type: str = ""

    def to_dict(self, ground_truth):
        x_key = self.id + '_x'
        y_key = self.id + '_y'

        x_ground_truth = ground_truth.loc[x_key, 'data_series']
        y_ground_truth = ground_truth.loc[y_key, 'data_series']
        x_numeric_flag = True if isinstance(x_ground_truth[0], float) else False
        y_numeric_flag = True if isinstance(y_ground_truth[0], float) else False

        try:
            if x_numeric_flag:
                all_x = [float(x) for x in self.x_series]
            else:
                all_x = [x for x in self.x_series]
        except:
            all_x = [0.0]
        try:
            if y_numeric_flag:
                all_y = [float(y) for y in self.y_series]
            else:
                all_y = [y for y in self.y_series]
        except:
            all_y = [0.0]

        x_value = (all_x, self.chart_type)
        y_value = (all_y, self.chart_type)
        return x_key, x_value, y_key, y_value

    def to_csv_result(self):
        x_data_series = ";".join(self.x_series)
        y_data_check = False
        for y_data in self.y_series:
            if not is_value(y_data):
                y_data_check = True
                break
        if y_data_check:
            y_data_series = ['1']
        else:
            y_data_series = ";".join(self.y_series)
        return [f'{self.id}_x', x_data_series, self.chart_type], \
            [f'{self.id}_y', y_data_series, self.chart_type]

    def __str__(self):
        return str(self.to_csv_result())

    @staticmethod
    def get_id(filepath):
        file_name_with_extension = os.path.basename(filepath)
        file_name_without_extension, _ = os.path.splitext(file_name_with_extension)
        return file_name_without_extension

    @staticmethod
    def default_result(filepath):
        r = ReadResult()
        r.id = ReadResult().get_id(filepath)
        r.chart_type = "vertical_bar"
        r.x_series.append('0.0')
        r.x_series.append('1.0')
        r.y_series.append('0.0')
        r.y_series.append('1.0')
        return r
