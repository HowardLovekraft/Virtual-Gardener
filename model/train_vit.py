from enum import Enum
from datetime import datetime
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from pandas import read_csv
import torch    
from torchvision.models.vision_transformer import vit_b_32
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import decode_image
import torchvision.transforms as transforms

from env_loader import DATASET, DEVICE, vit_epoch_amount


class DatasetSplit(Enum):
    TRAIN = 'train'
    TEST = 'test'
    VALID = 'valid'


class WhoWeAreDataset(Dataset):
    def __init__(self, annotations_dir: Path, img_dir: Path, 
                 split: DatasetSplit, transform: Optional[Callable] = None):
        
        annotations_path = Path(annotations_dir, split.value, 'annotations.csv')
        self.data_dir = Path(img_dir, split.value)
        self.annotations = read_csv(annotations_path, delimiter=',',
                                    names=['Image_Path', 'Disease'],
                                    dtype={0: object, 1: np.int32})
        self.transform = transform
 
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.annotations.iloc[idx, 0]
        label = self.annotations.iloc[idx, 1]
        image = decode_image(img_path).float()
        
        if self.transform:
            image = self.transform(image)

        image /= 255.

        return image, label


def main():
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.type(torch.LongTensor).to(DEVICE)
            # Zero gradients for batch
            optimizer.zero_grad()

            # Make predictions for batch
            outputs = eff_net(inputs)

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

    CWD = os.getcwd()
    architecture_name = 'vit'
    DATASET_ABS_PATH = Path(CWD, DATASET)
    # transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


    dataset_path = Path(DATASET_ABS_PATH)
    data_path = Path(dataset_path, 'data')
    annotations_path = Path(dataset_path, 'annotations')

    train_set = WhoWeAreDataset(annotations_path, data_path, split=DatasetSplit.TRAIN)
    valid_set = WhoWeAreDataset(annotations_path, data_path, split=DatasetSplit.VALID)
    test_set = WhoWeAreDataset(annotations_path, data_path, split=DatasetSplit.TEST)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)


    eff_net = vit_b_32()
    eff_net.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(eff_net.parameters())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    Path(CWD, 'runs').mkdir(exist_ok=True)
    writer = SummaryWriter(Path(CWD, 'runs', f'fashion_trainer_{timestamp}'))
    epoch_number = 0
    best_vloss = 1_000_000.

   
    for epoch in range(vit_epoch_amount):
        print(f'EPOCH {epoch_number + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        eff_net.train(True)
        print('\tStart training...')
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        eff_net.eval()
        print('\t Train is done. Start validating...')

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(valid_loader):
                vinputs = vinputs.to(DEVICE)
                vlabels = vlabels.type(torch.LongTensor).to(DEVICE)
                
                voutputs = eff_net(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

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
            torch.save(eff_net.state_dict(), model_path)

        epoch_number += 1


    # Загрузка весов
    # saved_model = EfficientNet()
    # saved_model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    main()