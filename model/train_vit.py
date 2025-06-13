from datetime import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision.models.vision_transformer import vit_b_32
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from model.env_loader import DATASET, DEVICE, vit_epoch_amount
from model.vision_transformer.utils import DatasetSplit, WhoWeAreDataset


def main():
    global vit_weights_path, annotations_path, data_path
    global transform

    def train_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, (inputs, labels) in enumerate(train_loader):
            inputs: torch.Tensor = inputs.to(DEVICE)
            labels = labels.type(torch.LongTensor).to(DEVICE)
            # Zero gradients for batch
            optimizer.zero_grad()

            # Make predictions for batch
            outputs: torch.Tensor = model(inputs)

            # Compute the loss and gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 20 == 19:
                last_loss = running_loss / 4 # loss per batch
                print('batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    def validate_epoch() -> tuple[float, int]:
        running_vloss = 0.0
        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(valid_loader):
                    vinputs = vinputs.to(DEVICE)
                    vlabels = vlabels.type(torch.LongTensor).to(DEVICE)
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss
        return running_vloss, len(valid_loader)


    CWD = os.getcwd()
    architecture_name = 'vit'

    DATASET_ABS_PATH = Path(CWD, DATASET)
    dataset_path = Path(DATASET_ABS_PATH)
    data_path = Path(dataset_path, 'data')
    annotations_path = Path(dataset_path, 'annotations')
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = WhoWeAreDataset(annotations_path, data_path, split=DatasetSplit.TRAIN, transform=transform)
    valid_set = WhoWeAreDataset(annotations_path, data_path, split=DatasetSplit.VALID, transform=transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=4)

    with open(Path('metric_analysis', 'labels_tags.json'), 'r') as file:
        labels_tags = json.load(file)


    model = vit_b_32(
        image_size=512,
        num_classes=len(labels_tags)    
    )
    model.to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path(CWD, 'runs').mkdir(exist_ok=True)
    writer = SummaryWriter(Path(CWD, 'runs', f'fashion_trainer_{timestamp}'))
    
    epoch_number = 0
    best_vloss = 1_000_000.
    for epoch in range(vit_epoch_amount):
        print(f'EPOCH {epoch_number + 1}:')

        model.train(True)
        print('Start training...')
        avg_loss = train_epoch(epoch_number, writer)

        
        model.eval()
        print('Train is done. Start validating...')
        running_vloss, val_size = validate_epoch()

        avg_vloss = running_vloss / val_size
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = Path(CWD, 'runs', f'{architecture_name}_{timestamp}_{epoch_number}')
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    vit_weights_path = Path(CWD, f'vit_b_32_{timestamp}.pt')
    model_scripted = torch.jit.script(model)
    model_scripted.save(vit_weights_path)


if __name__ == '__main__':
    main()