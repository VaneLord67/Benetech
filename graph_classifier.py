from enum import Enum


class GraphType(Enum):
    DOT = "dot"
    LINE = "line"
    SCATTER = "scatter"
    HORIZONTAL_BAR = "horizontal_bar"
    VERTICAL_BAR = "vertical_bar"


class GraphClassifier:
    def __init__(self):
        pass

    def classify(self, graph_path: str) -> GraphType:
        with open(graph_path, 'r') as file:
            return GraphType.BAR
