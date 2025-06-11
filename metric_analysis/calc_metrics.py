import json
import os
from pathlib import Path
import sys
from typing import NamedTuple, TypeAlias

from pandas import read_csv
import sklearn
import sklearn.metrics
import torch    
from torchvision.models.efficientnet import efficientnet_v2_s
from torchvision.io import decode_image

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

YOLO8_WEIGHTS = Path('metric_analysis', 'weights', 'yolo8_50epochs.pt')
YOLO11_WEIGHTS = Path('metric_analysis', 'weights', 'yolo11_50epochs.pt')


path = Path(DATASET, 'annotations', 'test')
labels_tags = None
with open('labels_tags.json', 'r') as file:
    labels_tags = json.load(file)
test_set = read_csv(Path(path, 'annotations.csv'), delimiter=',')
test_images = test_set.iloc[:, 0].tolist()
test_labels = test_set.iloc[:, 1].tolist()

# model_a = YOLO(YOLO8_WEIGHTS, task='classify')
# model_b = YOLO(YOLO11_WEIGHTS, task='classify')
model_c = efficientnet_v2_s()
model_c.load_state_dict(torch.load(Path(os.getcwd(), 'model', 'efficient_net_20250606_100413_0')),
                        assign=True)
model_c.to(DEVICE)


predictions_a = []
predictions_b = []
predictions_c = []
result_a = None


for image in test_images:
    # result_a = model_a.predict(image, device=DEVICE)[0]  # list with predicts
    # result_b = model_b.predict(image, device=DEVICE)[0]  # list with predicts
    print(image)
    result_c = model_c(decode_image(image).to(DEVICE).float().unsqueeze(0))
    # predictions_a.append(result_a.probs.top1)
    # predictions_b.append(result_b.probs.top1)
    predictions_c.append(result_c)
    

# stat_a = sklearn.metrics.classification_report(test_labels, predictions_a, 
#                                       labels=list(labels_tags.values()), 
#                                       target_names=list(labels_tags.keys()), 
#                                       digits=3, output_dict=False, zero_division='warn')

# stat_b = sklearn.metrics.classification_report(test_labels, predictions_b, 
#                                       labels=list(labels_tags.values()), 
#                                       target_names=list(labels_tags.keys()), 
#                                       digits=3, output_dict=False, zero_division='warn')

stat_c = sklearn.metrics.classification_report(test_labels, predictions_c, 
                                      labels=list(labels_tags.values()), 
                                      target_names=list(labels_tags.keys()), 
                                      digits=3, output_dict=False, zero_division='warn')


print(stat_a)
print(stat_b)
print(stat_c)