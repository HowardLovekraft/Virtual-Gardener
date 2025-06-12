from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
import torch
from torchvision.models.vision_transformer import vit_b_32
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from env_loader import DATASET, DEVICE, vit_epoch_amount
from vision_transformer.utils import DatasetSplit, WhoWeAreDataset


vit_weights_path = None


def main():
    def train_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.type(torch.LongTensor).to(DEVICE)
            # Zero gradients for batch
            optimizer.zero_grad()

            # Make predictions for batch
            outputs = model(inputs)

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

    def validate_epoch() -> float:
        running_vloss = 0.0
        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(valid_loader):
                    vinputs = vinputs.to(DEVICE)
                    vlabels = vlabels.type(torch.LongTensor).to(DEVICE)
                    
                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss
        return running_vloss

    CWD = os.getcwd()
    architecture_name = 'vit'

    DATASET_ABS_PATH = Path(CWD, DATASET)
    dataset_path = Path(DATASET_ABS_PATH)
    data_path = Path(dataset_path, 'data')
    annotations_path = Path(dataset_path, 'annotations')

    train_set = WhoWeAreDataset(annotations_path, data_path, split=DatasetSplit.TRAIN)
    valid_set = WhoWeAreDataset(annotations_path, data_path, split=DatasetSplit.VALID)
    test_set = WhoWeAreDataset(annotations_path, data_path, split=DatasetSplit.TEST)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

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
        running_vloss = validate_epoch()

        avg_vloss = running_vloss / (i+1)
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
            model_path = Path(CWD, f'{architecture_name}_{timestamp}_{epoch_number}')
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    vit_weights_path = Path('metric_analysis', 'weights', f'vit_b_32_{timestamp}')
    model_scripted = torch.jit.script(model)
    model_scripted.save(vit_weights_path)

    # Загрузка весов
    # saved_model = EfficientNet()
    # saved_model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    main()