import os
import json
from pathlib import Path
from typing import Final


CWD: Final[str] = os.getcwd()
DATASET_ABS_PATH: Final[str] = Path(CWD, 'papka', 'data')
ANNOTATIONS_PATH: Final[str] = Path(CWD, 'papka', 'annotations')

splits = os.listdir(DATASET_ABS_PATH)  # [test, train, valid]
classes = set()


# Find all labels in all splits to enumerate them
for split in splits:
    split_path = Path(DATASET_ABS_PATH, split)
    labels_ = [dir for dir in os.listdir(split_path)
               if os.path.isdir(Path(split_path, dir)) == True]
    for label in labels_:
        classes.add(label)


labels_numbers = {label: num for (num, label) in enumerate(classes)}
with open(Path('metric_analysis', 'labels_tags.json'), 'w') as file:
    json.dump(labels_numbers, file)

for split in splits:
    annotations = []
    split_path = Path(DATASET_ABS_PATH, split)
    labels_ = [dir for dir in os.listdir(split_path)
               if os.path.isdir(Path(split_path, dir)) == True]
    for label in labels_:
        label_path = Path(split_path, label)
        images = os.listdir(label_path)
        annotations.extend([str(Path(split_path, label, img_name)) + f", {labels_numbers[label]}\n" 
                            for img_name in images])

    with open(Path(ANNOTATIONS_PATH, split, 'annotations.csv'), 'w', encoding='utf-8') as file:
        writer = file.writelines(annotations)