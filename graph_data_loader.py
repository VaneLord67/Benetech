import os


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
