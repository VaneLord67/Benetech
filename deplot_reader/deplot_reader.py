import torch
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, AutoProcessor

import env
from abstract_graph_reader import AbstractGraphReader
from read_result import ReadResult, is_value

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Pix2StructForConditionalGeneration.from_pretrained(env.ROOT_PATH + "huggingface/models--google--deplot",
                                                           is_vqa=False).to(device)
processor = AutoProcessor.from_pretrained(env.ROOT_PATH + "huggingface/models--google--deplot", is_vqa=False)


# processor = AutoProcessor.from_pretrained(env.ROOT_PATH + "huggingface/deplot-finetune",
#                                           is_vqa=False, padding=True, truncation=True)
# processor.tokenizer.add_tokens(train.new_tokens)
# model = Pix2StructForConditionalGeneration.from_pretrained(env.ROOT_PATH + "huggingface/deplot-finetune",
#                                                            is_vqa=False).to(device)
# model.resize_token_embeddings(len(processor.tokenizer))

def count_duplicate_elements(lst):
    unique_elements = set(lst)
    if len(unique_elements) < len(lst):
        duplicate_elements = len(lst) - len(unique_elements)
        return duplicate_elements
    else:
        return 0


def check_if_failed(read_result: ReadResult):
    if len(read_result.x_series) == 1 and read_result.x_series[0] == "0.0":
        return True
    cnt = count_duplicate_elements(read_result.x_series)
    if cnt > 2:
        return True
    return False


class DeplotReader(AbstractGraphReader):
    def __init__(self, chart_type):
        self.chart_type = chart_type

    def read_graph(self, filepath) -> ReadResult:
        r = ReadResult()
        r.id = ReadResult.get_id(filepath)
        r.chart_type = self.chart_type
        image = Image.open(filepath)
        inputs = processor(images=image, text="Generate underlying data table of the figure below:",
                           return_tensors="pt").to(device)
        predictions = model.generate(**inputs, max_new_tokens=512).to(device)
        result = processor.decode(predictions[0], skip_special_tokens=True)
        elements = result.split("<0x0A>")
        elements.reverse()

        x_datas = []
        y_datas = []
        for element in elements:
            if '|' in element:
                extract_datas = element.split("|")
                if len(extract_datas) == 2:
                    if not is_value(extract_datas[1]):
                        break
                    x_datas.append(extract_datas[0].strip())
                    y_datas.append(extract_datas[1].strip())
        x_datas.reverse()
        y_datas.reverse()
        if len(x_datas) == 0:
            x_datas.append('0.0')
        if len(y_datas) == 0:
            y_datas.append('0.0')
        r.x_series = x_datas
        r.y_series = y_datas
        return r


if __name__ == '__main__':
    print(DeplotReader("vertical_bar").read_graph(env.DATASET_PATH + "train/images/00d54d817e51.jpg"))
