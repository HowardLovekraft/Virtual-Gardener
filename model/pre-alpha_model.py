from dotenv import load_dotenv
from os import getenv
from pathlib import PurePath
from shutil import move
from ultralytics import YOLO
from torch import cuda
from typing import Final


load_dotenv('./venv.env')

PATH_TO_YAML: Final[str] = PurePath("./YOLO_files/yolov8-cls.yaml")

DATASET_PATH_VAR_NAME: Final[str] = 'TMP_DATASET'
PATH_TO_DATASET: Final[str] = getenv(DATASET_PATH_VAR_NAME)
if PATH_TO_DATASET is not None:
    TMP_DATASET: Final[str] = PurePath(PATH_TO_DATASET)
else:
    raise Exception(f"Path to dataset in '{DATASET_PATH_VAR_NAME}' variable is not initialized.\nCheck your .env file.")

DEVICE: Final[str] = 'cuda' if cuda.is_available() else 'cpu'
print(DEVICE)

# YOLOv8; train from scratch
model = YOLO(PATH_TO_YAML)
train_results = model.train(data=TMP_DATASET, epochs=50, imgsz=256, device=DEVICE)

metrics = model.val()
print(metrics.top5)

model_path = model.export(format='onnx', device=0)

move(model_path, '/data')