if __name__ == '__main__':
    import sys
    sys.path.insert(1, '/kaggle/input/python-code')
    import os
    from tqdm import tqdm
    from example_reader.example_reader import ExampleReader
    from abstract_graph_reader import AbstractGraphReader
    from csv_processor import CSVProcessor
    from graph_classifier.graph_classifier import GraphClassifier, GraphType
    from graph_data_loader import GraphDataLoader
    from vertical_bar_reader.vertical_bar_reader import VerticalBarReader
    from read_result import ReadResult
    from env import *
    try:
        # 读取数据
        graph_data_loader = GraphDataLoader()
        # 产生csv结果文件
        csv_processor = CSVProcessor(os.getcwd() + "/submission.csv")
        # 分类图表
        graph_classifier = GraphClassifier(ROOT_PATH + "graph_classifier/graph_classifier.pth")
        file_generator = graph_data_loader.get_graph_filenames(DATASET_PATH + "test/images")
        for file_path in tqdm(file_generator):
            read_result: ReadResult
            try:
                graph_type = graph_classifier.classify(file_path)
                graph_reader: AbstractGraphReader
                if graph_type == GraphType.VERTICAL_BAR:
                    graph_reader = VerticalBarReader(ROOT_PATH + 'vertical_bar_reader/best.pt')
                # elif graph_type == GraphType.HORIZONTAL_BAR:
                #     graph_reader = HorizontalBarReader()
                # elif graph_type == GraphType.DOT:
                #     graph_reader = DotReader()
                # elif graph_type == GraphType.LINE:
                #     graph_reader = LineReader()
                # elif graph_type == GraphType.SCATTER:
                #     graph_reader = ScatterReader()
                else:
                    graph_reader = ExampleReader()
                # 产生结果
                read_result = graph_reader.read_graph(file_path)
                # print(read_result)
            except Exception as e:
                print(e)
                read_result = ReadResult.default_result(file_path)
            # 写入csv
            csv_processor.append(read_result)
        # csv文件检查
        csv_processor.file_check()
    except Exception as e:
        print(e)
