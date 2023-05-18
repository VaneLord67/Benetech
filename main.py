from abstract_graph_reader import AbstractGraphReader
from csv_processor import CSVProcessor
from dot_reader.dot_reader import DotReader
from graph_classifier.graph_classifier import GraphClassifier, GraphType
from graph_data_loader import GraphDataLoader
from horizontal_bar_reader.horizontal_bar_reader import HorizontalBarReader
from line_reader.line_reader import LineReader
from scatter_reader.scatter_reader import ScatterReader
from vertical_bar_reader.vertical_bar_reader import VerticalBarReader

if __name__ == '__main__':
    # 读取数据
    graph_data_loader = GraphDataLoader()
    # 产生csv结果文件
    csv_processor = CSVProcessor("test.csv")
    # 分类图表
    graph_classifier = GraphClassifier()
    graph_filenames = graph_data_loader.get_graph_filenames()
    graph_path = ""
    graph_type = graph_classifier.classify(graph_path)
    graph_reader: AbstractGraphReader
    if graph_type == GraphType.VERTICAL_BAR:
        graph_reader = VerticalBarReader()
    elif graph_type == GraphType.HORIZONTAL_BAR:
        graph_reader = HorizontalBarReader()
    elif graph_type == GraphType.DOT:
        graph_reader = DotReader()
    elif graph_type == GraphType.LINE:
        graph_reader = LineReader()
    else:
        graph_reader = ScatterReader()
    # 产生结果
    read_result = graph_reader.read_graph(graph_path)
    # 写入csv
    csv_processor.append(read_result)
    # 关闭csv文件
    csv_processor.close()
