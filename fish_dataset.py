import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import ImageEnhance, Image


class FishDataset(Dataset):
    def __init__(self, dataframe, classes_names, tf=None):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.transforms = tf
        self.samples = {s: i for i, s in enumerate(classes_names)}

    def __getitem__(self, index: int):
        # Retriving image id and records from df
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        # Loading Image
        frame_path = list(records['frame_path'])[0]
        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        sharpner = ImageEnhance.Sharpness(Image.fromarray(image))
        image = sharpner.enhance(8)

        boxes = np.zeros((records.shape[0], 4))
        boxes[:, 0:4] = records[['x1', 'y1', 'x2', 'y2']].values

        # Applying Transforms
        if self.transforms is not None:
            image = self.transforms(image)
            boxes = self.transforms(boxes).squeeze(0)
        targets = torch.tensor([self.samples[label] for label in records.label])

        return {'img': image, 'annot': boxes, 'label': targets}

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    @staticmethod
    def collater(data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'] for s in data]
        labels = [s['label'] for s in data]

        widths = [int(s.shape[1]) for s in imgs]
        heights = [int(s.shape[2]) for s in imgs]
        batch_size = len(imgs)

        max_width = np.max(widths)  # .max()
        max_height = np.max(heights)  # .max()

        padded_imgs = torch.zeros(batch_size, 3, max_width, max_height)

        for i in range(batch_size):
            img = imgs[i]
            padded_imgs[i, :] = img

        max_num_annots = max(annot.shape[0] for annot in annots)
        if max_num_annots > 0:

            annot_padded = torch.ones((len(annots), max_num_annots, 4)) * -1

            if max_num_annots > 0:
                for idx, annot in enumerate(annots):
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0], :] = annot
        else:
            annot_padded = torch.ones((len(annots), 1, 4)) * -1

        return {'img': padded_imgs, 'annot': annot_padded, 'label': labels}


class FishDataLoaders:
    _MODES = ['train', 'val', 'test']

    def __init__(self, frames_dir: str, train_path: str = 'train.csv', val_path: str = 'validate.csv',
                 test_path: str = 'test.csv'):
        train_df = pd.read_csv(os.path.join(frames_dir, train_path))
        val_df = pd.read_csv(os.path.join(frames_dir, val_path))
        test_df = pd.read_csv(os.path.join(frames_dir, test_path))
        classes_names = set(train_df['label']) | set(val_df['label']) | set(test_df['label'])
        transform = transforms.ToTensor()
        self._image_datasets = {x: FishDataset(dataframe=data, classes_names=classes_names, tf=transform)
                                for data, x in zip([train_df, val_df, test_df], FishDataLoaders._MODES)}
        self._dataloaders = {
            x: torch.utils.data.DataLoader(self._image_datasets[x], batch_size=8 if x == 'train' else 1,
                                           shuffle=True, num_workers=2, collate_fn=FishDataset.collater)
            for x in ['train', 'val', 'test']}
        self._dataset_sizes = {x: len(self._image_datasets[x]) for x in FishDataLoaders._MODES}
        self._label_to_species = dict(zip(train_df.label, train_df.species))
        self._label_to_species.update(dict(zip(val_df.label, val_df.species)))
        self._label_to_species.update(dict(zip(test_df.label, test_df.species)))
        self._idx_to_species = {i: self._label_to_species[x] for i, x in enumerate(classes_names)}
        self._mode = 'train'
        self._num_classes = len(classes_names)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        if mode not in FishDataLoaders._MODES:
            mode = 'train'
        self._mode = mode

    def get_db_size(self) -> int:
        return self._dataset_sizes[self._mode]

    @property
    def dataloader(self) -> torch.utils.data.DataLoader:
        return self._dataloaders[self._mode]

    @property
    def datasets(self):
        return self._image_datasets[self._mode]

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def dataset_sizes(self):
        return self._dataset_sizes

    @property
    def idx_to_species(self):
        return self._idx_to_species
