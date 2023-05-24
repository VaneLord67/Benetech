from easyocr import easyocr

DEV = "dev"
COMMIT = "commit"

DEBUG_MODE = False
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