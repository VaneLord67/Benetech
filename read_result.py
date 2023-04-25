from typing import List


class ReadResult:
    def __init__(self):
        # 注意，ReadResult中处理的都是字符串，如果某些坐标数据是数字浮点类型，要先转成字符串。
        self.id: str = ""
        self.x_series: List[str] = []
        self.y_series: List[str] = []
        self.chart_type: str = ""

    def toCSVResult(self):
        x_data_series = ";".join(self.x_series)
        y_data_series = ";".join(self.y_series)
        return [f'{self.id}_x', x_data_series, self.chart_type], \
               [f'{self.id}_y', y_data_series, self.chart_type]
