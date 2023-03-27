import os
import json
import numpy as np
from glob import glob

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

from Common.utils import split_filename, load_file


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class SyntheicsDataseLoader2D(Dataset):
    def __init__(self, cfg, phase):
        self.cfg = cfg
        self.main_modal = cfg.data.main_modal
        self.seq_modal = cfg.data.seq_modal
        self.data_augment = cfg.data_augment.enable

        self.data_list = cfg.data.data_file
        self.sample_list = glob(os.path.join(self.data_list, '*'))

        if phase != 'train' or not self.data_augment:
            self.dict_weak_transform = transforms.Compose([
                    CenterCropD(self.patch_size),
                    NormalizationD(self.window_level),
                    CreateOnehotLabel(self.num_class)])
            self.dict_strong_transform = transforms.Compose([
                    CenterCropD(self.patch_size),
                    NormalizationD(self.window_level),
                    CreateOnehotLabel(self.num_class)])
        else:
            self.dict_weak_transform = transforms.Compose([
                RandomRotFlipD(),
                ResizeD(scales=(0.8, 1.2), num_class=self.num_class),
                RandomCropD(self.patch_size),
                RandomBrightnessAdditiveD(labels=[i for i in range(1, self.num_class + 1)],
                                          additive_range=(-200, 200)),
                NormalizationD(self.window_level),
                CreateOnehotLabel(self.num_class)])
            self.dict_strong_transform = transforms.Compose([
                RandomRotFlipD(),
                ResizeD(scales=(0.8, 1.2), num_class=self.num_class),
                RandomCropD(self.patch_size),
                RandomBrightnessAdditiveD(labels=[i for i in range(1, self.num_class+1)],
                                          additive_range=(-200, 200)),
                NormalizationMinMaxD(self.window_level),
                ColorJitter(),
                NoiseJitter(),
                PaintingJitter(),
                NormalizationD([-1, 1]),
                CreateOnehotLabel(self.num_class)])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data = self.sample_list[idx]

        data_main_modal_path = glob(os.path.join(data, f'*{self.main_modal}*'))[0]
        data_seq_modal_path = glob(os.path.join(data, f'*{self.seq_modal}*'))[0]

        data_main_modal_image = load_file(data_main_modal_path)
        data_seq_modal_image = load_file(data_seq_modal_path)

        return data_main_modal_image, data_seq_modal_image

