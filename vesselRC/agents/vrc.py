"""
Main Agent for CondenseNet
"""
import numpy as np

from tqdm import tqdm
import shutil

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from agents.base import BaseAgent
from datasets import VRC_data
from utils.train_utils import adjust_learning_rate
from graphs.models import VRC_model
from tqdm  import tqdm
from skimage import measure
import trimesh
import os
import utils

cudnn.benchmark = True

class VRCAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # point could encoder
        self.encoder = VRC_model.Encoder(config)
        # occpany decoder
        self.decoder = VRC_model.Decoder(config)
        # model gather
        #self.model =[self.encoder,self.decoder]
        self.model = nn.Sequential(self.encoder,self.decoder)
        
        # Create instance from the loss
        # [T] self.loss = CrossEntropyLoss()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        # Create instance from the optimizer
        self.optimizer = torch.optim.Adam([{'params': self.encoder.parameters()},
                                        {'params':self.decoder.parameters()},],
                                         lr=self.config.learning_rate,
                                         betas=(self.config.beta1,self.config.beta2)
                                         )

        # data_loader
        self.data_loader = VRC_data.VRC_DataLoader(self.config)
        #dataset
        self.dataset = self.data_loader.dataset
        
        # initialize my counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0
        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Operation will be on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.config.seed)
            self.logger.info("Operation will be on *****CPU***** ")

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        #self.loss = self.loss.to(self.device)
        # Model Loading from the latest checkpoint if not found start from scratch.

        if self.config.use_pre_trian:
            self.load_checkpoint(self.config.checkpoint_file)

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='CondenseNet')
        
        # set for macubes
        self.overlap = self.config.overlap
        self.conservative = True
        self.interp = False
    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        self.logger.info("save model in " + self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):

        filename = self.config.checkpoint_dir + filename
        try:
            if self.config.use_Localimplicit_decoder_pretrain:
                pretrained_dict = torch.load(self.config.LocalGrid_pretrian_file)
                model_dict = self.decoder.state_dict()
                update_dict = {k[8:] : v for k, v in pretrained_dict.items()}
                model_dict.update(update_dict)
                self.decoder.load_state_dict(model_dict)
                self.logger.info("Checkpoint loaded successfully from '{}'\n"
                                .format(self.config.LocalGrid_pretrian_file))
            else:
                self.logger.info("Loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
                self.current_epoch = checkpoint['epoch']
                self.current_iteration = checkpoint['iteration']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])

                self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                                .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:
                # [t] self.train()
                self.test_one_data(0,'{}debug.ply'.format('test'))

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()

            # valid_acc = self.validate()
            # is_best = valid_acc > self.best_valid_acc
            # if is_best:
            #     self.best_valid_acc = valid_acc
            if epoch %10 == 9:
                self.save_checkpoint('{}model.pth.tar'.format(epoch), True)
                self.test_one_data(0,'{}debug.ply'.format(epoch))

    def decode(self,grid,pts):
        if self.model.training:
            self.interp = False
        else:
            self.interp = True
        lat, weights, xloc = self.grid_interpolation(grid, pts) # pipelines\lig\models\layers.py:67
        # -> lat:[bs,pt_size,8,latent_size], weights:[bs,pt_size,8],xloc:[bs,pt_size,8,3]
        # -> xloc is the (x-xc), the local coordinates relate to the local grid
        # [t] xloc = xloc * self.x_location_max
        if self.config.method == "linear":
            input_features = torch.cat([xloc, lat], dim=3)  # bs*npoints*nneighbors*c
            values = self.decoder(input_features)
            if self.interp:
                values = (values * weights.unsqueeze(3)).sum(dim=2)  # bs*npoints*1
            else:
                values = (values, weights)
        else:
            # nearest neighbor
            bs, npoints, nneighbors, c = lat.size()
            nearest_neighbor_idxs = weights.max(dim=2, keepdim=True)[1]
            lat = torch.gather(lat, dim=2, index=nearest_neighbor_idxs.unsqueeze(3).expand(bs, npoints, 1,
                                                                                           c))  # bs*npoints*1*c
            lat = lat.squeeze(2)  # bs*npoints*c
            xloc = torch.gather(xloc, dim=2, index=nearest_neighbor_idxs.unsqueeze(3).expand(bs, npoints, 1, 3))
            xloc = xloc.squeeze(2)  # bs*npoints*3
            input_features = torch.cat([xloc, lat], dim=2)
            values = self.decoder(input_features)
        return values
    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize tqdm
        # tqdm_batch = tqdm(self.data_loader.train_loader,
        #                   desc="Epoch-{}-".format(self.current_epoch))
        tqdm_batch = self.data_loader.train_loader
        # Set the model to be in training mode
        # Initialize your average meters
        self.model.train()
        current_batch = 0
        for b_grid_points, b_occ_idx, b_grid_shape, b_sdf_points in tqdm_batch:
            # convert the data
            # [T] pass
            # encoder the point_cloud
            grid_points_lens =[ len(grid_points) for grid_points in b_grid_points]
            b_grid_points = torch.cat(b_grid_points,dim=0)
            b_grid_points = b_grid_points.to(self.device)

            b_grid_latent = self.encoder(b_grid_points) #[bs*occ_size,latent_size]
            
            # padding the latent_grid with zero_latent
            pre_i = 0
            b_latent_grid = None
            b_sample_sdf_points = None
            for i, gi in enumerate(grid_points_lens):
                if b_latent_grid == None:
                    b_latent_grid = self.padded_latent(occ_idxs=b_occ_idx[i],occ_latent=b_grid_latent[pre_i:pre_i+gi],grid_shape=b_grid_shape).unsqueeze(0)
                    b_sample_sdf_points = self.sample_sdf_points(b_sdf_points[i], grid_shape = b_grid_shape,occ_idxs= b_occ_idx[i]).unsqueeze(0)
                else:
                    t_latent_grid = self.padded_latent(occ_idxs=b_occ_idx[i],occ_latent=b_grid_latent[pre_i:pre_i+gi],grid_shape=b_grid_shape).unsqueeze(0)
                    b_latent_grid = torch.cat([b_latent_grid,t_latent_grid],dim=0)
                    t_sample_sdf_points = self.sample_sdf_points(b_sdf_points[i], grid_shape = b_grid_shape,occ_idxs= b_occ_idx[i]).unsqueeze(0)
                    b_sample_sdf_points = torch.cat([b_sample_sdf_points,t_sample_sdf_points],dim=0)
                pre_i = gi
            
            # if len(grid_points_lens) > 1: # batch_size > 1
            #     lat = torch.cat(b_lat,dim=0).to(self.device)
            #     xloc = torch.cat(b_xloc,dim=0).to(self.device)
            #     input_features = torch.cat([xloc, lat], dim=3)
            #     weights = torch.cat(b_weights,dim=0).to(self.device)
            #     point_val_samples = torch.cat(b_sample_sdf_points,dim=0)[:,3]
            #     point_val_samples = torch.sign(point_val_samples).to(self.device)
            #     point_val_samples = (point_val_samples+1)/2  # 0 / 1
            # else:
            #     lat = b_lat[0].to(self.device)
            #     xloc = b_xloc[0].to(self.device)
            #     input_features = torch.cat([xloc, lat], dim=3)
            #     weights = b_weights[0].to(self.device)
            #     point_val_samples = b_sample_sdf_points[0][:,3]
            #     point_val_samples = torch.sign(point_val_samples).to(self.device)
            #     point_val_samples = point_val_samples[None,:,None]
            #     point_val_samples = (point_val_samples+1)/2  # 0 / 1
            # decoder the sdf
            b_latent_grid = b_latent_grid.to(self.device)
            b_sample_sdf_points = b_sample_sdf_points.to(self.device)
            pred , weights = self.decode(b_latent_grid,b_sample_sdf_points[:,:,:3])
            pred_interp = (pred * weights.unsqueeze(3)).sum(dim=2, keepdim=True)
            pred = torch.cat([pred, pred_interp], dim=2)  # 1*npoints*9*1
            # sign value
            point_val_samples = b_sample_sdf_points[:,:,3]
            point_val_samples = torch.sign(point_val_samples)
            point_val_samples = (point_val_samples+1)/2  # 0 / 1
            point_val_samples = point_val_samples[:,:,None,None]
            
            binary_labels = point_val_samples.expand(*pred.size())  # 1*npoints*9*1
            pred_flatten = pred.reshape(-1, 1)
            binary_labels = binary_labels.reshape(-1, 1)
            # loss
            self.optimizer.zero_grad()
            loss = self.loss_fn(pred_flatten, binary_labels).mean()
            loss.backward()
            self.optimizer.step()
            #tqdm_batch.set_description("Processing %s" % loss.item())
            print(loss)
    def test_one_data(self,idx,output_ply):
        self.model.eval()
        xmin=(-1.,-1.,-1.)
        xmax=(1.,1.,1.)
        b_grid_points,occ_idx,grid_shape,_ = self.dataset[idx]
        true_shape = ((np.array(grid_shape) - 1) / (2.0 if self.config.overlap else 1.0)).astype(np.int32)
        output_grid_shape = list(self.config.res_per_part * true_shape)
        output_grid, xyz = self.get_eval_grid(xmin=xmin,
                                              xmax=xmax,
                                              output_grid_shape=output_grid_shape)
        occ_mask = self.occ_idx_mask(occ_idx.numpy(), grid_shape.numpy())
        eval_points, out_mask = self.get_eval_inputs(xyz, xmin, occ_mask)
        eval_points = torch.from_numpy(eval_points).to(self.device)
        grid_points = b_grid_points.to(self.device)
        occ_latent_grid = self.encoder(grid_points)
        latent_grid = self.padded_latent(occ_idxs=occ_idx,occ_latent=occ_latent_grid, grid_shape= grid_shape).unsqueeze(0)
        output_grid = self.generate_occ_grid(latent_grid, eval_points, output_grid, out_mask)
        output_grid = output_grid.reshape(*output_grid_shape)
        v, f, _, _ = measure.marching_cubes(output_grid, 0.75)  # logits==0.5
        v *= (self.config.part_size / float(self.config.res_per_part) * (np.array(output_grid.shape, dtype=np.float32) /
                                                           (np.array(output_grid.shape, dtype=np.float32) - 1)))
        v += xmin
        mesh = trimesh.Trimesh(v, f)
        mesh.export(output_ply)
        self.logger.info("successfully matching cube the {}-th data, save in {}".format(idx,output_ply))
    def validate(self):
        pass
    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()

    def padded_latent(self, occ_idxs, occ_latent, grid_shape):
        '''
            padded the latent grid with zero latent
        '''
        noccupied = occ_idxs.shape[0] # -> valid crop size
        si, sj, sk = grid_shape
        occ_idxs_flatten = occ_idxs[:, 0] * (sj * sk) + occ_idxs[:, 1] * sk + occ_idxs[:, 2]  # bs*npoints
        latent_grid = torch.zeros((si * sj * sk), self.config.latent_size).type(torch.cuda.FloatTensor)
        latent_grid[occ_idxs_flatten] = occ_latent
        latent_grid = latent_grid.reshape(si, sj, sk, self.config.latent_size)
        return latent_grid
    def occ_idx_mask(self, occ_idx, grid_shape):
        dense = np.zeros(grid_shape, dtype=np.bool).ravel()
        # -> the ravel() would flatten the tensor without duplicate the source
        # -> the flatten() would flatten the tensor with return duplicate
        occ_idx_f = (occ_idx[:, 0] * grid_shape[1] * grid_shape[2] + occ_idx[:, 1] * grid_shape[2] + occ_idx[:, 2])
        dense[occ_idx_f] = True
        dense = np.reshape(dense, grid_shape)
        return dense
    def uniform_sample(self,pts,sample_num):
        random_pos = (torch.rand(sample_num) * pts.shape[0]).long()
        # torch.rand 为0~1之间的均匀采样，sample_number 表示要采点的个数
        sample_pts = torch.index_select(pts, 0, random_pos)
        return sample_pts
    def sample_sdf_points(self,pts,
                        uniform_random_select = False,
                        uniform_random_sample_number = 10000,
                        grid_shape=None,occ_idxs=None,
                        empty_sample_number = 1000):
        npoints, _ = pts.shape
        if uniform_random_select:
            return self.uniform_sample(pts, uniform_random_sample_number)
        elif grid_shape != None and occ_idxs != None:
            npoints, _ = pts.shape
            xmin = torch.tensor([-1., -1., -1.])
            xmax = torch.tensor([1., 1., 1.])
            size = torch.FloatTensor(list(grid_shape))
            cube_size = (xmax -xmin) / (size - 1)

            # normalize coords for interpolation
            pts_t = pts[:,:3] - torch.tensor([-1., -1., -1.]) #(npoints,4)
            ind0 = (pts_t[:,:3] / cube_size.reshape([1,-1])).floor()  # grid index (npoints, 3)

            # get 8 neighbors
            offset = torch.Tensor([0, 1])
            grid_x, grid_y, grid_z = torch.meshgrid(*tuple([offset] * 3))
            neighbor_offsets = torch.stack([grid_x, grid_y, grid_z], dim=-1)
            neighbor_offsets = neighbor_offsets.reshape(-1, 3)  # 8*3
            nneighbors = neighbor_offsets.shape[0]
            neighbor_offsets = neighbor_offsets.type(torch.FloatTensor)  # shape==(8, 3)

            # get neighbor 8 latent codes
            neighbor_indices = ind0.unsqueeze(1) + neighbor_offsets[None, :, :]  # (npoints, 8, 3)
            neighbor_indices = neighbor_indices.type(torch.LongTensor)
            neighbor_indices = neighbor_indices.reshape(-1, 3)  # (npoints*8, 3)
            d, h, w = neighbor_indices[:, 0], neighbor_indices[:, 1], neighbor_indices[:, 2]

            occ_mask = self.occ_idx_mask(occ_idxs.numpy(),grid_shape.numpy())
            neighbor_indices_occ_mask = occ_mask[d, h, w] # (npoints*8,)
            neighbor_indices_occ_mask = neighbor_indices_occ_mask.reshape(-1,8) #(npoints,8)
            pts_mask = torch.zeros(npoints).type(torch.bool)
            for i in range(8):
                pts_mask = pts_mask | neighbor_indices_occ_mask[:,i]
            occ_grid_pts = pts[pts_mask]
            occ_grid_pts = self.uniform_sample(occ_grid_pts,uniform_random_sample_number - empty_sample_number)
            no_occ_grid_pts = pts[~pts_mask]
            select_no_occ_grid_pts= self.uniform_sample(no_occ_grid_pts,empty_sample_number)
            sample_pts = torch.cat((occ_grid_pts,select_no_occ_grid_pts),dim=0)
            return sample_pts
    def get_eval_grid(self, xmin, xmax, output_grid_shape):
        """Initialize the eval output grid and its corresponding grid points.

        Args:
            xmin (numpy array): [3], minimum xyz values of the entire space.
            xmax (numpy array): [3], maximum xyz values of the entire space.
            output_grid_shape (list): [3], latent grid shape.
        Returns:
             output_grid (numpy array): [d*h*w] output grid sdf values.
             xyz (numpy array): [d*h*w, 3] grid point xyz coordinates.
        """
        # setup grid
        eps = 1e-6
        l = [np.linspace(xmin[i] + eps, xmax[i] - eps, output_grid_shape[i]) for i in range(3)]
        xyz = np.stack(np.meshgrid(l[0], l[1], l[2], indexing='ij'), axis=-1).astype(np.float32)

        output_grid = np.ones(output_grid_shape, dtype=np.float32)
        xyz = xyz.reshape(-1, 3)
        output_grid = output_grid.reshape(-1)

        return output_grid, xyz
    def get_eval_inputs(self, xyz, xmin, occ_mask):
        """Gathers the points within the grids that any/all of its 8 neighbors
        contains points.

        If self.conservative is True, gathers the points within the grids that any of its 8 neighbors
        contains points.
        If self.conservative is False, gathers the points within the grids that all of its 8 neighbors
        contains points.
        Returns the points need to be evaluate and the mask of the points and the output grid.

        Args:
            xyz (numpy array): [h*w*d, 3]
            xmin (numpy array): [3] minimum value of the entire space.
            occ_mask (numpy array): latent grid occupancy mask.
        Returns:
            eval_points (numpy array): [neval, 3], points to be evaluated.
            out_mask (numpy array): [h*w*d], 0 1 value eval mask of the final sdf grid.
        """
        mask = occ_mask.astype(np.bool)
        if self.overlap:
            mask = np.stack([
                mask[:-1, :-1, :-1], mask[:-1, :-1, 1:], mask[:-1, 1:, :-1], mask[:-1, 1:, 1:], mask[1:, :-1, :-1],
                mask[1:, :-1, 1:], mask[1:, 1:, :-1], mask[1:, 1:, 1:]
            ],
                            axis=-1)
            if self.conservative:
                mask = np.any(mask, axis=-1)
            else:
                mask = np.all(mask, axis=-1)

        g = np.stack(np.meshgrid(np.arange(mask.shape[0]),
                                 np.arange(mask.shape[1]),
                                 np.arange(mask.shape[2]),
                                 indexing='ij'),
                     axis=-1).reshape(-1, 3)
        g = g[:, 0] * (mask.shape[1] * mask.shape[2]) + g[:, 1] * mask.shape[2] + g[:, 2]
        g_valid = g[mask.ravel()]  # valid grid index

        if self.overlap:
            ijk = np.floor((xyz - xmin) / self.config.part_size * 2).astype(np.int32)
        else:
            ijk = np.floor((xyz - xmin + 0.5 * self.config.part_size) / self.config.part_size).astype(np.int32)
        ijk_idx = (ijk[:, 0] * (mask.shape[1] * mask.shape[2]) + ijk[:, 1] * mask.shape[2] + ijk[:, 2])
        out_mask = np.isin(ijk_idx, g_valid)
        eval_points = xyz[out_mask]
        return eval_points, out_mask
    def generate_occ_grid(self, latent_grid, eval_points, output_grid, out_mask):
        """Gets the final output occ grid.

        Args:
            latent_grid (tensor): [1, *grid_shape, latent_size], optimized latent grid.
            eval_points (tensor): [neval, 3], points to be evaluated.
            output_grid (numpy array): [d*h*w], final output occ grid.
            out_mask (numpy array): [d*h*w], mask indicating the grids evaluated.
        Returns:
            output_grid (numpy array): [d*h*w], final output occ grid flattened.
        """

        split = int(np.ceil(eval_points.shape[0] / self.config.points_batch))
        occ_val_list = []
        self.model.eval()
        with torch.no_grad():
            for s in range(split):
                sid = s * self.config.points_batch
                eid = min((s + 1) * self.config.points_batch, eval_points.shape[0])
                eval_points_slice = eval_points[sid:eid, :]
                occ_vals = self.decode(latent_grid, eval_points_slice.unsqueeze(0))
                occ_vals = occ_vals.squeeze(0).squeeze(1).cpu().numpy()
                occ_val_list.append(occ_vals)
        occ_vals = np.concatenate(occ_val_list, axis=0)
        output_grid[out_mask] = occ_vals
        return output_grid
    def grid_interpolation(self,grid,pts):
        bs, npoints, _ = pts.shape
        # xmin = self.xmin.reshape([1, 1, -1])
        # xmax = self.xmax.reshape([1, 1, -1])
        
        xmin = torch.tensor([-1.,-1.,-1.]).to(self.device)
        xmax = torch.tensor([1.,1.,1.]).to(self.device)
        size = torch.cuda.FloatTensor(list(grid.shape[1:-1]))
        cube_size = (xmax-xmin) / (size - 1)
        # !!! there maybe a problem with scale the sphere to cube
        # normalize coords for interpolation
        #pts = (pts - xmin) / (xmax - xmin)  # normalize to 0 ~ 1
        
        pts = pts - xmin
        #pts = pts.clamp(min=1e-6, max=1 - 1e-6)
        ind0 = (pts / cube_size.reshape([1, 1, -1])).floor()  # grid index (bs, npoints, 3)

        # get 8 neighbors
        offset = torch.Tensor([0, 1])
        grid_x, grid_y, grid_z = torch.meshgrid(*tuple([offset] * 3))
        neighbor_offsets = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        neighbor_offsets = neighbor_offsets.reshape(-1, 3)  # 8*3
        nneighbors = neighbor_offsets.shape[0]
        neighbor_offsets = neighbor_offsets.type(torch.cuda.FloatTensor)  # shape==(8, 3)

        # get neighbor 8 latent codes
        neighbor_indices = ind0.unsqueeze(2) + neighbor_offsets[None, None, :, :]  # (bs, npoints, 8, 3)
        neighbor_indices = neighbor_indices.type(torch.cuda.LongTensor)
        neighbor_indices = neighbor_indices.reshape(bs, -1, 3)  # (bs, npoints*8, 3)
        # ！！！ for remove the grid sdf_points which no a latent
            
        d, h, w = neighbor_indices[:, :, 0], neighbor_indices[:, :, 1], neighbor_indices[:, :, 2]  # (bs, npoints*8)
        batch_idxs = torch.arange(bs).type(torch.cuda.LongTensor)
        batch_idxs = batch_idxs.unsqueeze(1).expand(bs, npoints * nneighbors)  # bs, 8*npoints
        # ???
        lat = grid[batch_idxs, d, h, w, :]  # bs, (npoints*8), c
        lat = lat.reshape(bs, npoints, nneighbors, -1)

        # get the tri-linear interpolation weights for each point
        xyz0 = ind0 * cube_size.reshape([1, 1, -1])  # (bs, npoints, 3)
        xyz0_expand = xyz0.unsqueeze(2).expand(bs, npoints, nneighbors, 3)  # (bs, npoints, nneighbors, 3)
        xyz_neighbors = xyz0_expand + neighbor_offsets[None, None, :, :] * cube_size

        neighbor_offsets_oppo = 1 - neighbor_offsets
        xyz_neighbors_oppo = xyz0.unsqueeze(2) + neighbor_offsets_oppo[None,
                                                                       None, :, :] * cube_size  # bs, npoints, 8, 3
        dxyz = (pts.unsqueeze(2) - xyz_neighbors_oppo).abs() / cube_size
        weight = dxyz[:, :, :, 0] * dxyz[:, :, :, 1] * dxyz[:, :, :, 2]

        # relative coordinates inside the grid (-1 ~ 1, e.g. [0~1,0~1,0~1] for min vertex, [-1~0,-1~0,-1~0] for max vertex)
        xloc = (pts.unsqueeze(2) - xyz_neighbors) / cube_size[None, None, None, :]
        # -> xloc is the (x-xc), x the local coordinates relate to the local grid
        return lat, weight, xloc
