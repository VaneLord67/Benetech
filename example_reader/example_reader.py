from abstract_graph_reader import AbstractGraphReader
from read_result import ReadResult


# 继承AbstractGraphReader，并实现read_picture
class ExampleReader(AbstractGraphReader):
    def __init__(self):
        self.read_result = ReadResult()

    def get_read_result(self):
        return self.read_result

    def read_graph(self, filepath):
        with open(filepath, 'r') as file:
            # 处理文件后写入到read_result中即可。
            self.read_result.id = "example_id"
            self.read_result.chart_type = "example_type"
            self.read_result.x_series.append("example_x")
            self.read_result.y_series.append("example_y")
