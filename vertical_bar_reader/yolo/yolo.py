import cv2
from ultralytics import YOLO

from vertical_bar_reader.vertical_bar_reader import on_mouse


def train(model_path, epochs):
    # Load a model
    model = YOLO(model_path)  # load a pretrained model (recommended for training)
    # model = YOLO('./yolo/yolov8.yaml')
    # Train the model
    model.train(data='./yolo/dataset.yaml', epochs=epochs, imgsz=500)


def test(model_path, id):
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(model_path)
    # Perform object detection on an image using the model
    img = cv2.imread(f'./dataset/train/images/{id}.jpg')
    res = model(img)
    res_plotted = res[0].plot()
    boxes = res[0].boxes
    for box in boxes:
        print(box.xyxy[0][0].item())
    cv2.imshow("result", res_plotted)
    cv2.setMouseCallback('result', on_mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test('./runs/detect/train13/weights/best.pt', "e0895ead1808")
    # train('./runs/detect/train12/weights/last.pt', epochs=120)
