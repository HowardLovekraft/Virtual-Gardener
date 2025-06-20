from os import getenv
from pathlib import Path
from torch import cuda
from typing import Final


# Загрузка переменной с директорией датасета
DATASET_PATH_VAR_NAME: Final[str] = 'DATASET_DIR'
PATH_TO_DATASET = getenv(DATASET_PATH_VAR_NAME)
if PATH_TO_DATASET is None:
    raise EnvironmentError(
        f"Path to dataset in '{DATASET_PATH_VAR_NAME}' variable is not initialized.\nCheck your docker-compose.yml."
    )
DATASET: Final[Path] = Path(PATH_TO_DATASET)


# Загрузка переменной с файлом конфигурации модели
YAML_PATH_VAR_NAME: Final[str] = 'MODEL_YAML'
PATH_TO_YAML = getenv(YAML_PATH_VAR_NAME)
if PATH_TO_YAML is None:
    raise EnvironmentError(
        f"Path to config in {YAML_PATH_VAR_NAME} variable is not initialized.\nCheck your docker-compose.yml."
    )
YAML_FILE: Final[Path] = Path(PATH_TO_YAML)


# Загрузка переменной с файлом весов модели
MODEL_PATH_VAR_NAME: Final[str] = 'PRETRAINED_MODEL'
PATH_TO_MODEL = getenv(MODEL_PATH_VAR_NAME)
if PATH_TO_MODEL is None:
    raise EnvironmentError(
        f"Path to model in '{MODEL_PATH_VAR_NAME}' variable is not initialized.\nCheck your docker-compose.yml."
    )
WEIGHTS_PATH: Final[Path] = Path(PATH_TO_MODEL)


# Загрузка устройства, на котором будет работать модель - CPU или GPU(CUDA)
DEVICE: Final[str] = 'cuda' if cuda.is_available() else 'cpu'
print(f"You'll train and use model on {DEVICE.upper()}!")