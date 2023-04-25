from csv_processor import CSVProcessor
from example_reader.example_reader import ExampleReader
from read_result import ReadResult

if __name__ == '__main__':
    csv_processor = CSVProcessor("test.csv")
    example_reader = ExampleReader()
    example_reader.read_picture("dataset/train/images/0a0a78fc8d65.jpg")
    csv_processor.append(example_reader.get_read_result())
    csv_processor.close()