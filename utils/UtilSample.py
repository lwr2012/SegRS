import os
import numpy as np
from osgeo import gdal
from skimage import io

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from utils.UtilAttribute import AttrDict
from utils.UtilBase import get_files, get_training_augmentation
from utils.UtilRegister import SampleLoader


class ImageDataset(Dataset):
    def __init__(self, **kwargs):
        self.config = AttrDict(**kwargs)
        self.train_img_files, self.train_mask_files = [], []
        self.val_img_files, self.val_mask_files = [], []
        self.split_dataset(
            self.config.get('train_img_dir'),
            self.config.get('train_mask_dir'),
            self.config.get('val_img_dir'),
            self.config.get('val_mask_dir')
        )

    def split_dataset(self, train_img_dir, train_mask_dir, val_img_dir, val_mask_dir):

        train_names = get_files(
            train_img_dir,
            extend=self.config.get('extend', '.png')
        )

        val_names = get_files(
            val_img_dir,
            extend=self.config.get('extend', '.png')
        )

        for name in train_names:
            self.train_img_files.append(os.path.join(train_img_dir, name))
            self.train_mask_files.append(os.path.join(train_mask_dir, name))

        for name in val_names:
            self.val_img_files.append(os.path.join(val_img_dir, name))
            self.val_mask_files.append(os.path.join(val_mask_dir, name))

    @staticmethod
    def _read_img(image_path):
        dataset = gdal.Open(image_path)
        data = dataset.ReadAsArray()
        data = np.transpose(data, [1, 2, 0])
        # data = np.load(image_path)
        data = data.astype(np.float32)
        return data

    @staticmethod
    def _normalize(img):
        return img / 255.0

    def _get_data(self, img_files, mask_files, idx):
        img = self._read_img(img_files[idx])
        mask = self._read_img(mask_files[idx])
        return img, mask

    def __getitem__(self, idx):

        if self.config.get('training',True):
            img, mask = self._get_data(self.train_img_files, self.train_mask_files, idx)
            if self.config.get('augmentation', True):
                transformed = get_training_augmentation()(image=img, mask=mask)
                img, mask = transformed['image'],transformed['mask']
        else:
            img, mask = self._get_data(self.val_img_files, self.val_mask_files, idx)

        img = self._normalize(img)

        img = np.transpose(img, [2, 0, 1])

        mask = np.transpose(mask, [2, 0, 1])

        return img, mask

    def __len__(self):
        if self.config.get('training',True):
            return len(self.train_img_files)
        else:
            return len(self.val_img_files)


@SampleLoader.register('Dataset')
class ImageLoader:
    def __init__(self, **kwargs):
        # img_dir,mask_dir,training=True,augmentation=True,train_ratio=0.9,batch_size=10,extend='.tif'
        self.dataset = ImageDataset(**kwargs)

        self.g = torch.Generator()
        self.g.manual_seed(self.dataset.config.get('seed', 666666))
        self.batch_sampler = BatchSampler(
            RandomSampler(
                range(len(self.dataset)),
                replacement=self.dataset.config.get('replacement', False),
                num_samples=self.dataset.config.get('num_samples', None),
                generator=self.g
            ),
            batch_size=self.dataset.config.get('batch_size', 10),
            drop_last=self.dataset.config.get('drop_last', False)
        )

    def __call__(self, **kwargs):
        data_loader = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            **kwargs
        )

        train_len = len(self.dataset.train_img_files)
        batch_size = self.dataset.config.get('batch_size', 8)

        data_loader.total_epoch = int(np.ceil(train_len / batch_size))
        return data_loader

