from shutil import move

from ultralytics import YOLO

from env_loader import DATASET, DEVICE, YOLO11_CONFIG, yolo_epoch_amount


# YOLO 11; train from scratch
model = YOLO(YOLO11_CONFIG, task='classify')
train_results = model.train(data=DATASET, epochs=yolo_epoch_amount, imgsz=256, device=DEVICE)

# Валидация
metrics = model.val()
print(metrics.top5)

# Экспорт модели
weights_path = model.export(format='onnx', device=DEVICE)
move(weights_path, '/data/yolo11-weights.onnx')
