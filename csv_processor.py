import csv

from read_result import ReadResult


class CSVProcessor:
    def __init__(self, csv_file_name):
        header = ['id', 'data_series', 'chart_type']  # 表头
        file = open(f'output/{csv_file_name}', 'w', newline='')
        writer = csv.writer(file)
        self.file = file
        self.csv_file_writer = writer
        writer.writerow(header)

    def append(self, read_result: ReadResult):
        result_x, result_y = read_result.toCSVResult()
        self.csv_file_writer.writerow(result_x)
        self.csv_file_writer.writerow(result_y)

    def close(self):
        self.file.close()
