from abc import ABC, abstractmethod


# 这是用@abstractmethod装饰器表示的一个接口，指定了需要实现的函数：read_picture。
# 需要按接口定义来写实现类，具体的例子在example_reader中的example_reader.py可以看到如何实现这个接口。
class AbstractGraphReader(ABC):
    @abstractmethod
    def read_picture(self, filepath):
        pass

    @abstractmethod
    def get_read_result(self):
        pass