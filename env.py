from paddleocr import PaddleOCR

from easyocr import easyocr

DEV = "dev"
COMMIT = "commit"

DEBUG_MODE = True
ENV = DEV


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def is_commit_env():
    return ENV == COMMIT


DATASET_PATH = "/kaggle/input/benetech-making-graphs-accessible/" \
    if is_commit_env() else "dataset/"

ROOT_PATH = "/kaggle/input/python-code/" if is_commit_env() else ""

READER = easyocr.Reader(['en'], model_storage_directory=ROOT_PATH + "easyocr/model", download_enabled=True)
PADDLE_READER = PaddleOCR(use_angle_cls=False, lang="en", show_log=False,
                          cls_model_dir=ROOT_PATH + 'paddle_ocr_models/ch_ppocr_mobile_v2.0_cls_infer',
                          rec_model_dir=ROOT_PATH + 'paddle_ocr_models/en_PP-OCRv3_rec_infer',
                          # det_model_dir=ROOT_PATH + 'paddle_ocr_models/en_PP-OCRv3_det_infer'
                          det_model_dir=ROOT_PATH + 'paddle_ocr_models/finetune_det'
                          )
