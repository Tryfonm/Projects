from tqdm import trange
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Autoencoder(nn.Module):
    def __init__(self, bottleneck_dimension=10):
        super().__init__()
        self.d = bottleneck_dimension
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, self.d ** 2),
            nn.ReLU(),
            nn.Linear(self.d ** 2, self.d)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.d, self.d ** 2),
            nn.ReLU(),
            nn.Linear(self.d ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels=3, bottleneck_dimension=10):
        super().__init__()
        self.input_channels = input_channels
        self.d = bottleneck_dimension
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=16, kernel_size=4, stride=2, padding=1),  # 1 x 28 x 28 --> 16 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0),  # 16 x 14 x 14 --> 32 x 6 x 6
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, self.d)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.d, self.d ** 2),
            nn.ReLU(),
            nn.Linear(self.d ** 2, 1152),
            Reshape(-1, 32, 6, 6),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0),  # 32 x 6 x 6 --> 16 x 14 x 14
            nn.ConvTranspose2d(in_channels=16, out_channels=self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16 x 14 x 14 --> 3 x 28 x 28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DenoisingAutoencoder(nn.Module):
    def __init__(self, bottleneck_dimension=10):
        super().__init__()
        self.d = bottleneck_dimension

        self.dropout = nn.Dropout(p=0.50)

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(
            nn.Linear(400, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
