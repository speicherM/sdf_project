"""
Mnist tutorial main model
"""
import torch.nn as nn
import torch.nn.functional as F
from ..weights_initializer import weights_init
import torch

class Decoder(nn.Module):
    """ImNet layer py-torch implementation."""

    def __init__(self, config):
        """Initialization.

        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          num_filters: int, width of the second to last layer.
          activation: activation function.
        """
        super(Decoder, self).__init__()
        self.config = config
        self.dim = self.config.point_dim
        self.in_features = self.config.IM_Net_in_features
        self.dimz = self.dim + self.in_features
        self.out_features = self.config.IM_Net_out_features
        self.num_filters = self.config.IM_Net_num_filters
        self.activ = nn.LeakyReLU(0.2)
        self.fc0 = nn.Linear(self.dimz, self.num_filters * 16)
        self.fc1 = nn.Linear(self.dimz + self.num_filters * 16, self.num_filters * 8)
        self.fc2 = nn.Linear(self.dimz + self.num_filters * 8, self.num_filters * 4)
        self.fc3 = nn.Linear(self.dimz + self.num_filters * 4, self.num_filters * 2)
        self.fc4 = nn.Linear(self.dimz + self.num_filters * 2, self.num_filters * 1)
        self.fc5 = nn.Linear(self.num_filters * 1, self.out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    def forward(self, x):
        """Forward method.

        Args:
          x: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          x_: output through this layer.
        """
        x_ = x
        for dense in self.fc[:4]:
            x_ = self.activ(dense(x_))
            x_ = torch.cat([x_, x], dim=-1)
        x_ = self.activ(self.fc4(x_))
        x_ = self.fc5(x_)
        return x_

class PartFeatSampler(nn.Module):

    def __init__(self, feature_size, probabilistic=True):
        super(PartFeatSampler, self).__init__()
        self.probabilistic = probabilistic

        self.mlp2mu = nn.Linear(feature_size, feature_size)
        self.mlp2var = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        mu = self.mlp2mu(x)

        if self.probabilistic:
            logvar = self.mlp2var(x)
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)

            kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            return torch.cat([eps.mul(std).add_(mu), kld], 1)
        else:
            return mu
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, self.config.latent_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(self.config.latent_size)

        self.sampler = PartFeatSampler(feature_size=self.config.latent_size, probabilistic=self.config.probabilistic)

    def forward(self, pc):
        
        net = pc.transpose(2, 1)
        net = torch.relu(self.bn1(self.conv1(net)))
        net = torch.relu(self.bn2(self.conv2(net)))
        net = torch.relu(self.bn3(self.conv3(net)))
        net = torch.relu(self.bn4(self.conv4(net)))

        net = net.max(dim=2)[0]
        net = self.sampler(net)

        return net