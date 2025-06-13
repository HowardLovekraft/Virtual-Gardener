import json
import os
from pathlib import Path
import sys
from typing import Callable, NamedTuple, TypeAlias

from pandas import read_csv
import sklearn
import sklearn.metrics
import torch
from torch.utils.data import DataLoader
from torchvision.io import decode_image
import torchvision.transforms as transforms

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.env_loader import DATASET, DEVICE
from model.train_vit import DatasetSplit, WhoWeAreDataset


Label: TypeAlias = str
LabelNum: TypeAlias = int

class Statistic(NamedTuple):
    TP: int
    TN: int
    FP: int
    FN: int


def test_model_performance(model: Callable):
    predictions: list[torch.types.Number] = []
    expectations: list[int] = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            outputs: torch.Tensor = model(inputs)
            predictions.append(outputs.argmax().item())
            expectations.append(labels.item())


    stats = sklearn.metrics.classification_report(expectations, predictions, 
                                        labels=list(labels_tags.values()), 
                                        target_names=list(labels_tags.keys()), 
                                        digits=3, output_dict=False, zero_division='warn')

    print(stats)


# YOLO8_WEIGHTS = Path('metric_analysis', 'weights', 'yolo8_50epochs.pt')
# YOLO11_WEIGHTS = Path('metric_analysis', 'weights', 'yolo11_50epochs.pt')

path = Path(DATASET, 'annotations', 'test')
labels_tags = None
with open(Path('metric_analysis', 'labels_tags.json'), 'r') as file:
    labels_tags = json.load(file)

CWD = os.getcwd()
DATASET_ABS_PATH = Path(CWD, DATASET)
dataset_path = Path(DATASET_ABS_PATH)
data_path = Path(dataset_path, 'data')
annotations_path = Path(dataset_path, 'annotations')
transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
vit_weights_path = Path(CWD, f'vit_b_32_{timestamp}.pt')

test_set = WhoWeAreDataset(annotations_path, data_path, split=DatasetSplit.VALID, transform=transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


model_c = torch.jit.load(vit_weights_path)
model_c.to(DEVICE)

test_model_performance(model_c)