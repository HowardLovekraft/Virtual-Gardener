from shutil import move
from ultralytics import YOLO

from env_loader import DATASET, DEVICE, YAML_FILE


# YOLO v8; train from scratch
model = YOLO(YAML_FILE, task='classify')
train_results = model.train(data=DATASET, epochs=2, imgsz=256, device=DEVICE)

# Валидация
metrics = model.val()
print(metrics.top5)

# Экспорт модели
weights_path = model.export(format='onnx', device=0)
move(weights_path, '/data/model.onnx')
