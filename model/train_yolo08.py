from shutil import move
from ultralytics import YOLO

from env_loader import DATASET, DEVICE, YOLO8_CONFIG, yolo_epoch_amount


# YOLO v8; train from scratch
model = YOLO(YOLO8_CONFIG, task='classify')
train_results = model.train(data=DATASET, epochs=yolo_epoch_amount, imgsz=256, device=DEVICE)

# Валидация
metrics = model.val()
print(metrics.top5)

# Экспорт модели
weights_path = model.export(format='onnx', device=DEVICE)
move(weights_path, '/data/model.onnx')
