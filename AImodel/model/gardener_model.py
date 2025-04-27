from os import makedirs
from pathlib import Path
from typing import TypedDict
from ultralytics import YOLO
import numpy as np
import cv2

from .env_loader import DEVICE, WEIGHTS_PATH


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

    def predict(self, image_data: bytes, image_size: int = 256) -> str:
        """
        Предсказывает болезнь растения по фотографии листвы и возвращает название класса.

        :param image_data: Байты файла фотографии листвы.

        :param image_size: Размер входного изображения. По умолчанию - 256*256 пикселей.

        :return prediction: Название предсказанного класса болезни растения.
        """
        # Преобразуем байты изображения в формат, понятный OpenCV и Ultralytics
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = self.model.predict(source=img,
                                        device=DEVICE,
                                        imgsz=image_size,
                                        verbose=False) # Отключаем verbose для чистоты вывода

        if results and results[0].probs is not None:
            top1_index = results[0].probs.top1
            prediction = self.class_names[int(top1_index)]
            return prediction
        else:
            return "Не удалось распознать класс" # Или другое сообщение об ошибке