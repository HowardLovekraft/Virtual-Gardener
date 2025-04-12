import json
from datetime import datetime
from os import makedirs
from pathlib import Path
from typing import TypedDict
from ultralytics import YOLO

from model.env_loader import DEVICE, WEIGHTS_PATH


class GardenerResponse(TypedDict):
    class_name: str


class VirtualGardener:
    """
    Класс, описывающий интерфейс взаимодействия с моделью Виртуального Садовника.

    :param path_to_save: Путь до директории, в которой будут сохраняться 
    предсказания модели.
    """
    def __init__(self, path_to_save: str) -> None:
        self.model = YOLO(WEIGHTS_PATH, task='classify')
        self.class_names = self.model.names
        self.save_dir = Path(path_to_save)
        # Создает директорию для предиктов, если её нет
        makedirs(path_to_save, exist_ok=True)

    def predict(self, path_to_image: str | Path, image_size: int = 256) -> Path:
        """
        Предсказывает болезнь растения по фотографии листвы и сохраняет ответ в `.json`.

        :param path_to_image: Путь до файла фотографии листвы.

        :param image_size: Размер входного изображения. По умолчанию - 256*256 пикселей.

        :return json_path: Путь до JSON-файла, содержащего предсказанный класс болезни растения.
        Название файла - время обращения к модели.
        """
        dt = str(hash(path_to_image))
        results = self.model.predict(source=path_to_image, 
                                     device=DEVICE, 
                                     imgsz=image_size)
        prediction = self.class_names[int(results[0].probs.top1)]
        response: GardenerResponse = {"class_name": prediction}

        json_path = self.save_dir / (dt + '.json')
        with open(json_path, mode='w', encoding='utf-8') as json_response:
            json.dump(response, json_response)
        return json_path

