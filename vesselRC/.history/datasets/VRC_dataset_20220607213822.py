import os

import numpy as np
import scipy.io as sio
import PIL
from PIL import Image
import torch

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as standard_transforms
import random
import utils.VRC_utils as extended_transforms
def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]

def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples

class VRCDataset(torch.utils.data.Dataset):
    def __init__(self,config):
        self.subsample = config.subsample
        self.mode = config.mode
        self.cour
        npyfiles = os.listdir(config.data_source)
        if self.mode == "train":
            if config.valid_data_random:
                pass
            else:
                self.npyfiles = npyfiles[0:len(npyfiles)//5*4]
        else:
            pass
        self.loaded_data = []
        for f in self.npyfiles:
            npz = np.load(f)
            pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
            neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
            self.loaded_data.append(
                [
                    pos_tensor[torch.randperm(pos_tensor.shape[0])],
                    neg_tensor[torch.randperm(neg_tensor.shape[0])],
                ]
            )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename =  os.path.join(self.soself.npyfiles[idx]
        unpack_sdf_samples(filename, self.subsample), idx


class VRCDataLoader:
    def __init__(self, config):
        self.config = config
        assert self.config.mode in ['train', 'test', 'random']

        mean_std = ([103.939, 116.779, 123.68], [1.0, 1.0, 1.0])

        self.input_transform = standard_transforms.Compose([
            standard_transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR),
            extended_transforms.FlipChannels(),
            standard_transforms.ToTensor(),
            standard_transforms.Lambda(lambda x: x.mul_(255)),
            standard_transforms.Normalize(*mean_std)
        ])

        self.target_transform = standard_transforms.Compose([
            standard_transforms.Resize((256, 256), interpolation=PIL.Image.NEAREST),
            extended_transforms.MaskToTensor()
        ])

        self.restore_transform = standard_transforms.Compose([
            extended_transforms.DeNormalize(*mean_std),
            standard_transforms.Lambda(lambda x: x.div_(255)),
            standard_transforms.ToPILImage(),
            extended_transforms.FlipChannels()
        ])

        self.visualize = standard_transforms.Compose([
            standard_transforms.Resize(400),
            standard_transforms.CenterCrop(400),
            standard_transforms.ToTensor()
        ])
        if self.config.mode == 'random':
            train_data = torch.randn(self.config.batch_size, self.config.input_channels, self.config.img_size,
                                     self.config.img_size)
            train_labels = torch.ones(self.config.batch_size, self.config.img_size, self.config.img_size).long()
            valid_data = train_data
            valid_labels = train_labels
            self.len_train_data = train_data.size()[0]
            self.len_valid_data = valid_data.size()[0]

            self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
            self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

            train = TensorDataset(train_data, train_labels)
            valid = TensorDataset(valid_data, valid_labels)

            self.train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
            self.valid_loader = DataLoader(valid, batch_size=config.batch_size, shuffle=False)

        elif self.config.mode == 'train':
            train_set = VOC('train', self.config.data_root,
                            transform=self.input_transform, target_transform=self.target_transform)
            valid_set = VOC('val', self.config.data_root,
                            transform=self.input_transform, target_transform=self.target_transform)

            self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False,
                                           num_workers=self.config.data_loader_workers,
                                           pin_memory=self.config.pin_memory)
            self.train_iterations = (len(train_set) + self.config.batch_size) // self.config.batch_size
            self.valid_iterations = (len(valid_set) + self.config.batch_size) // self.config.batch_size

        elif self.config.mode == 'test':
            test_set = VOC('test', self.config.data_root,
                           transform=self.input_transform, target_transform=self.target_transform)

            self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False,
                                          num_workers=self.config.data_loader_workers,
                                          pin_memory=self.config.pin_memory)
            self.test_iterations = (len(test_set) + self.config.batch_size) // self.config.batch_size

        else:
            raise Exception('Please choose a proper mode for data loading')

    def finalize(self):
        pass
