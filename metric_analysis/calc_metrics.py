import json
import os
from pathlib import Path
import sys
from typing import Callable, NamedTuple, TypeAlias

from pandas import read_csv
import sklearn
import sklearn.metrics
import torch
from torchvision.io import decode_image

from model.train_vit import vit_weights_path


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.env_loader import DATASET, DEVICE


Label: TypeAlias = str
LabelNum: TypeAlias = int

class Statistic(NamedTuple):
    TP: int
    TN: int
    FP: int
    FN: int


def test_model_performance(model: Callable):
    predictions = []
    with torch.no_grad():
        for image in test_images:
            result: torch.Tensor = model(decode_image(image).to(DEVICE).float().unsqueeze(0))
            print(result.shape)
            print('PREDICTED!')
            break
            predictions.append(result.to('cpu'))

    stats = sklearn.metrics.classification_report(test_labels, predictions, 
                                        labels=list(labels_tags.values()), 
                                        target_names=list(labels_tags.keys()), 
                                        digits=3, output_dict=False, zero_division='warn')

    print(stats)


YOLO8_WEIGHTS = Path('metric_analysis', 'weights', 'yolo8_50epochs.pt')
YOLO11_WEIGHTS = Path('metric_analysis', 'weights', 'yolo11_50epochs.pt')

path = Path(DATASET, 'annotations', 'test')
labels_tags = None
with open(Path('metric_analysis', 'labels_tags.json'), 'r') as file:
    labels_tags = json.load(file)

test_set = read_csv(Path(path, 'annotations.csv'), delimiter=',')
test_images = test_set.iloc[:, 0].tolist()
test_labels = test_set.iloc[:, 1].tolist()

# model_a = YOLO(YOLO8_WEIGHTS, task='classify')
# model_b = YOLO(YOLO11_WEIGHTS, task='classify')

model_c = torch.jit.load(vit_weights_path)
model_c.to(DEVICE)

test_model_performance(model_c)