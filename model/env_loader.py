from os import getenv
from pathlib import Path
import tomllib
from typing import Final

from dotenv import load_dotenv
from torch import cuda


# Загрузка конфига для обучения
yolo_epoch_amount = 0
effnet_epoch_amount = 0
vit_epoch_amount = 0
with open('.\\model\\train-config.toml', 'rb') as f:
    config = tomllib.load(f)
    yolo_epoch_amount = config['yolo_epoch_amount']
    effnet_epoch_amount = config['effnet_epoch_amount']
    vit_epoch_amount = config['vit_epoch_amount']


# Загрузка файла с переменными окружения
PATH_TO_ENV: Final[str] = '.\\model\\venv.env'
env_is_loaded = load_dotenv(PATH_TO_ENV)
if not env_is_loaded:
    raise EnvironmentError(
        f"Path to .env file '{PATH_TO_ENV}' doesn't exist.\nCheck existence of .env file and/or its' path."
    )


# Загрузка переменной с директорией датасета
DATASET_PATH_VAR: Final[str] = 'DATASET_DIR'
PATH_TO_DATASET = getenv(DATASET_PATH_VAR)
if PATH_TO_DATASET is None:
    raise EnvironmentError(
        f"Path to dataset in '{DATASET_PATH_VAR}' variable is not initialized.\nCheck your .env file."
    )
DATASET: Final[Path] = Path(PATH_TO_DATASET)


# Загрузка переменной с файлом конфигурации YOLO8
YOLO8_CONFIG_PATH_VAR: Final[str] = 'YOLO8_YAML'
PATH_TO_YOLO8_CONFIG = getenv(YOLO8_CONFIG_PATH_VAR)
if PATH_TO_YOLO8_CONFIG is None:
    raise EnvironmentError(
        f"Path to config in {PATH_TO_YOLO8_CONFIG} variable is not initialized.\nCheck your .env file."
    )
YOLO8_CONFIG: Final[Path] = Path(PATH_TO_YOLO8_CONFIG)


# Загрузка переменной с файлом конфигурации YOLO11
YOLO11_CONFIG_PATH_VAR: Final[str] = 'YOLO11_YAML'
PATH_TO_YOLO11_CONFIG = getenv(YOLO11_CONFIG_PATH_VAR)
if PATH_TO_YOLO11_CONFIG is None:
    raise EnvironmentError(
        f"Path to config in {PATH_TO_YOLO11_CONFIG} variable is not initialized.\nCheck your .env file."
    )
YOLO11_CONFIG: Final[Path] = Path(PATH_TO_YOLO11_CONFIG)


# Загрузка переменной с файлом весов модели
MODEL_PATH_VAR: Final[str] = 'PRETRAINED_MODEL'
PATH_TO_MODEL = getenv(MODEL_PATH_VAR)
if PATH_TO_MODEL is None:
    raise EnvironmentError(
        f"Path to model in '{MODEL_PATH_VAR}' variable is not initialized.\nCheck your .env file."
    )
WEIGHTS_PATH: Final[Path] = Path(PATH_TO_MODEL)


# Загрузка устройства, на котором будет работать модель - CPU или GPU(CUDA)
DEVICE: Final[str] = 'cuda' if cuda.is_available() else 'cpu'
print(f"You'll train and use model on {DEVICE.upper()}!")
