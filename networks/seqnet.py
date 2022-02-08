import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torchvision.models as models
from torch.utils.data import DataLoader, SubsetRandomSampler
import nuscene as dataset
import os
import math
from math import ceil
import faiss
from sklearn.cluster import KMeans
import h5py


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class seqNet(nn.Module):
    def __init__(self, inDims, outDims, seqL, w=5):

        super(seqNet, self).__init__()
        self.inDims = inDims
        self.outDims = outDims
        self.w = w
        self.conv = nn.Conv1d(inDims, outDims, kernel_size=self.w)

    def forward(self, x):                                                      # ([192, 10, 4096])

        if len(x.shape) < 3:
            x = x.unsqueeze(1)                                                 # convert [B,C] to [B,1,C]

        x = x.permute(0, 2, 1)                                                 # from [B,T,C] to [B,C,T]    ([192, 4096, 10])
        seqFt = self.conv(x)                                                   # ([192, 4096, 6])
        seqFt = torch.mean(seqFt, -1)                                          # ([192, 4096])

        return seqFt


class get_model(nn.Module):
    def __init__(self, opt, require_init=True):
        super(get_model, self).__init__()
        self.opt = opt
        outDims = opt.output_dim
        seqL = opt.seqLen
        self.seqNet = seqNet(opt.encoder_dim, outDims, seqL, opt.w)
        self.flatten = Flatten()
        self.l2norm = L2Norm()

    def forward(self, input):                                                  # ([32, 3, 4096, 1])
        input = torch.squeeze(input, 3)                                        # ([32, 3, 4096])
        feature = self.seqNet(input)                                           # ([32, 4096])
        feature = self.l2norm(self.flatten(feature))                           # ([32, 4096])

        return feature
