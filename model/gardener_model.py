import json
from datetime import datetime
from os import makedirs
from pathlib import Path
from typing import TypedDict

from PIL import Image
import torch
import torchvision.transforms as transforms
import io 

MODEL_BASE_DIR = Path(__file__).parent 

class GardenerResponse(TypedDict):
    class_name: str


class VirtualGardener:
    """
    Класс, описывающий интерфейс взаимодействия с моделью Виртуального Садовника.
    """
    vit_weights_path = MODEL_BASE_DIR / 'YOLO_files' / 'trained_models' / 'v0.1' / 'vit_b_32_20250613_163357.pt'
    labels_path = MODEL_BASE_DIR / 'labels_tags.json'

    def center_crop_pillow(self, img_pil: Image.Image, crop_width: int = 512, crop_height: int = 512) -> Image.Image:
        """
        Кропает изображение PIL до квадрата (например, 512x512).
        Принимает и возвращает объект PIL.Image.
        """
        width, height = img_pil.size
        left = (width - crop_width) / 2
        top = (height - crop_height) / 2
        right = (width + crop_width) / 2
        bottom = (height + crop_height) / 2
        new_img = img_pil.crop((left, top, right, bottom))
        return new_img

    def __init__(self, path_to_save: str) -> None:
        if not self.vit_weights_path.exists():
            raise FileNotFoundError(f"Веса модели не найдены по пути: {self.vit_weights_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Файл меток не найден по пути: {self.labels_path}")

        self.model = torch.jit.load(str(self.vit_weights_path), map_location='cpu') # Убедимся, что на CPU
        self.model.eval()

        with open(self.labels_path, 'r', encoding='utf-8') as file:
            self.class_indexes: dict = json.load(file)
            # В зависимости от формата labels_tags.json, эта строка может нуждаться в корректировке.
            # Если JSON: {"class_name_str": index_int}, то это ОК.
            # Если JSON: {"index_str": "class_name_str"}, то нужно {int(k): v for k, v in ...}
            # Предположим, что { "название_класса": номер_класса }
            self.class_names = {index: name for name, index in self.class_indexes.items()}
            print(f"DEBUG: Загружены class_names: {self.class_names}") # Отладочный вывод
            
        self.save_dir = Path(path_to_save)
        makedirs(path_to_save, exist_ok=True)
        
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image_data: bytes, image_size: int = 512) -> str:
        try:
            img_pil = Image.open(io.BytesIO(image_data))
            
            # Добавим вывод информации об изображении для отладки
            print(f"DEBUG: Изображение получено. Размер: {img_pil.size}, Формат: {img_pil.format}")

            cropped_img_pil = self.center_crop_pillow(img_pil, image_size, image_size)
            print(f"DEBUG: Изображение обрезано и изменено до {cropped_img_pil.size}")

            input_tensor = self.image_transform(cropped_img_pil)
            input_batch = input_tensor.unsqueeze(0) 
            print(f"DEBUG: Тензор готов. Форма: {input_batch.shape}, Тип: {input_batch.dtype}")


            with torch.no_grad():
                output = self.model(input_batch)
            
            print(f"DEBUG: Выход модели (логиты/вероятности): {output.shape}") # Отладочный вывод

            prediction_index = output.argmax(dim=1).item()
            prediction = self.class_names.get(prediction_index, "Болезнь не распознана (индекс не найден)") # Используем .get с дефолтом
            
            print(f"DEBUG: Предсказанный индекс: {prediction_index}") # Отладочный вывод
            print(f"DEBUG: Предсказанное имя класса из labels_tags.json: {prediction}") # Отладочный вывод
            return prediction 

        except Exception as e:
            print(f"ОШИБКА: Произошла ошибка во время предсказания: {e}") # Отладочный вывод
            return "Ошибка: Не удалось предсказать класс." 
        finally:
            pass