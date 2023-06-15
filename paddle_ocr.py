import cv2
from PIL import Image
from paddleocr import draw_ocr

import env

if __name__ == '__main__':
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = env.PADDLE_READER  # need to run only once to download and load model into memory
    # f64754927de3 数字
    # a8dcdd9d2b70 年份
    # 9fb13a928f91_x
    img_path = env.DATASET_PATH + "train/images/9fb13a928f91.jpg"
    image = cv2.imread(img_path)

    result = ocr.ocr(image, cls=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # 显示结果
    # 如果本地没有simfang.ttf，可以在doc/fonts目录下下载
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')