from shutil import move
from ultralytics import YOLO

from env_loader import DATASET, DEVICE, YOLO11_YAML


# YOLO 8; train from scratch
model = YOLO(YOLO11_YAML, task='classify')
train_results = model.train(data=DATASET, epochs=2, imgsz=256, device=DEVICE)

# Валидация
metrics = model.val()
print(metrics.top5)

# Экспорт модели
weights_path = model.export(format='onnx', device=DEVICE)
move(weights_path, '/data')
