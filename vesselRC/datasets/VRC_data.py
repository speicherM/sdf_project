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
from utils import point_utils as pu
def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]
class VRC_Dataset(torch.utils.data.Dataset):
    def __init__(self,config):
        self.config = config
        self.mode = config.mode
        self.data_source = config.data_source
        npyfiles = os.listdir(config.data_source)
        self.xmin = (-1.,-1.,-1.)
        self.xmax = (1.,1.,1.)
        if self.config.use_min_batch:
            if self.config.min_batch_random:
                pass
            else:
                self.npyfiles = [npyfiles[0]]
        else:
            if self.mode == "train":
                if config.valid_data_random:
                    pass
                else:
                    self.npyfiles = npyfiles#[0:len(npyfiles)//5*4]
            else:
                pass
        self.loaded_data = []
        for f in self.npyfiles:
            f = os.path.join(self.data_source,f)
            #sdf_and_pc = np.load(f, allow_pickle=True)
            self.loaded_data.append(f)

    def __len__(self):
        return len(self.loaded_data)

    def occ_idx_mask(self, occ_idx, grid_shape):
        dense = np.zeros(grid_shape, dtype=np.bool).ravel()
        # -> the ravel() would flatten the tensor without duplicate the source
        # -> the flatten() would flatten the tensor with return duplicate
        occ_idx_f = (occ_idx[:, 0] * grid_shape[1] * grid_shape[2] + occ_idx[:, 1] * grid_shape[2] + occ_idx[:, 2])
        dense[occ_idx_f] = True
        dense = np.reshape(dense, grid_shape)
        return dense

    def __getitem__(self, idx):
        sdf_and_pc =  np.load(self.loaded_data[idx], allow_pickle=True)
        sdf_points = sdf_and_pc['sdf_points'] #[sdf_points_size,4]
        pc = sdf_and_pc['point_cloud'] #[pc_size,3]
        grid_points, occ_idx, grid_shape = pu.np_get_occupied_idx(
            pc,
            xmin = self.xmin,
            xmax = self.xmax,
            crop_size = self.config.part_size,
            ntarget = self.config.ntarget,  # we do not require `point_crops` (i.e. `_` in returns), so we set it to 1
            overlap = self.config.overlap,
            normalize_crops = self.config.normalize_crops,
            return_shape=True)
        grid_points = torch.tensor(grid_points)
        occ_idx = torch.tensor(occ_idx)
        grid_shape = torch.tensor(grid_shape)
        sdf_points = torch.tensor(sdf_points)
        return grid_points,occ_idx,grid_shape,sdf_points

class VRC_DataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset = VRC_Dataset(config=config)
        self.data_len = len(self.dataset)
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.data_loader_workers,
            collate_fn = self.collate)
    def collate(self,datas):
        b_grid_points = []
        b_occ_idx = []
        b_grid_shape = []
        b_sdf_points = []
        for data in datas:
            b_grid_points.append(data[0])
            b_occ_idx.append(data[1])
            b_grid_shape = data[2]
            b_sdf_points.append(data[3])
        return b_grid_points, b_occ_idx, b_grid_shape, b_sdf_points
    def finalize(self):
        pass
