import os

from graph_classifier.train_graph_classify import get_chart_type_from_json


class GraphDataLoader:
    def __init__(self):
        pass

    @staticmethod
    def list_files(directory):
        files = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                files.append(filename)
        return files

    @staticmethod
    def get_graph_filenames():
        return GraphDataLoader.list_files("dataset/train/images")

    @staticmethod
    def get_annotations():
        return GraphDataLoader.list_files("dataset/train/annotations")

    @staticmethod
    def get_train_data():
        directory = "dataset/train/annotations"
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            chart_type = get_chart_type_from_json(path)
            graph_path = "dataset/train/images/" + filename[:-5] + ".jpg"
            yield graph_path, chart_type


