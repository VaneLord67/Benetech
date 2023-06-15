import cv2
from ultralytics import YOLO

import env

yolo_model = YOLO(env.ROOT_PATH + 'intersection/last.pt')


def get_intersection_yolo(img):
    res = yolo_model(img)
    if len(res) == 0:
        return None
    boxes = res[0].boxes
    if len(boxes) == 0:
        return None
    return round(boxes[0].xywh[0][0].item()), round(boxes[0].xywh[0][1].item())


if __name__ == '__main__':
    img = cv2.imread('dataset/train/images/0000ae6cbdb1.jpg')
    intersection = get_intersection_yolo(img)
    print(intersection)
