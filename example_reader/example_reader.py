from abstract_graph_reader import AbstractGraphReader
from read_result import ReadResult


# 继承AbstractGraphReader，并实现read_picture
class ExampleReader(AbstractGraphReader):
    def __init__(self):
        self.read_result = ReadResult()

    def read_graph(self, filepath) -> ReadResult:
        return ReadResult().default_result(filepath)
