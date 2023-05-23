import os
import pandas as pd
from read_result import ReadResult


class CSVProcessor:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        header = ['id', 'data_series', 'chart_type']  # 表头
        df = pd.DataFrame(columns=header)
        df.to_csv(csv_file_path, header=True, index=False)

    def append(self, read_result: ReadResult):
        df = pd.DataFrame({
            'id': [read_result.id + "_x", read_result.id + "_y"],
            'data_series': [";".join(read_result.x_series), ";".join(read_result.y_series)],
            'chart_type': ['vertical_bar', 'vertical_bar'],
        })
        df.to_csv(self.csv_file_path, index=False, header=False, mode='a')

    def file_check(self):
        if os.path.exists(self.csv_file_path):
            print("csv file exist")
        else:
            print("csv file not found")
