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
    def __init__(self, path_to_save: str) -> None:
        self.model = YOLO(WEIGHTS_PATH, task='classify')
        self.class_names = self.model.names
        self.save_dir = Path(path_to_save)
        makedirs(path_to_save, exist_ok=True)

    def predict(self, image_data: bytes, image_size: int = 256) -> str:
        temp_filename = f"temp_image_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
        temp_image_path = self.save_dir / temp_filename

        with open(temp_image_path, "wb") as f:
            f.write(image_data)

        try:
            results = self.model.predict(source=temp_image_path,
                                         device=DEVICE,
                                         imgsz=image_size)

            prediction = self.class_names[int(results[0].probs.top1)]

            return prediction

        finally:
            if temp_image_path.exists():
                temp_image_path.unlink()