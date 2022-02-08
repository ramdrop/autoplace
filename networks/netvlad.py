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
import argparse


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""
    def __init__(self, opt, num_clusters=64, dim=128, normalize_input=True, vladv2=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            vladv2 : bool
                If true, use vladv2 otherwise use vladv1
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def init_params(self, clsts, traindescs):                                       # (64, 512), (50000, 512)
        # TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)      # (64, 512)
            dots = np.dot(clstsAssign, traindescs.T)                                # (64, 50000) = (64, 512) * (512, 50000)^T
            dots.sort(0)                                                            # (64, 50000)
            dots = dots[::-1, :]  # sort, descending                                # (64, 50000)

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()  # 461
            self.centroids = nn.Parameter(torch.from_numpy(clsts))                  # ([64, 512])
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))   # ([64, 512, 1, 1])
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)  # TODO faiss?
            knn.fit(traindescs)
            del traindescs
            a = knn.kneighbors(clsts, 2)                                            # [(64, 2), (64, 2)] distances, indices
            b = a[1]
            dsSq = np.square(b)                                                     # (64, 2)
            # dsSq = np.square(knn.kneighbors(clsts, 2)[1])                             # (64, 2)
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
            self.conv.bias = nn.Parameter(-self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):                                           # ([32, 512, 9, 9])
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim # (B, D, N, 1)

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)  # ([32, 64, 81])
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)                               # (32, 512, 81)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)  # (32, 64, 512)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


def get_encoder(opt):
    # ===== build model
    pretrained = not opt.fromscratch
    opt.encoder_dim = 512
    encoder = models.vgg16(pretrained=pretrained)
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]

    if pretrained:
        # if using pretrained then only train conv5_1, conv5_2, and conv5_3
        for l in layers[:-5]:
            for p in l.parameters():
                # p.requires_grad = False
                pass

    layers.append(L2Norm())
    encoder = nn.Sequential(*layers)
    model = nn.Module()
    model.add_module('encoder', encoder)
    return model


class get_model(nn.Module):
    def __init__(self, opt, require_init=True):
        super(get_model, self).__init__()
        self.opt = opt
        self.require_init = require_init
        self.encoder = get_encoder(self.opt)
        self.netvlad = NetVLAD(opt, num_clusters=opt.num_clusters, dim=opt.encoder_dim)
        if opt.seqLen > 1:
            self.lstm = nn.LSTM(input_size=9216, hidden_size=self.opt.output_dim, num_layers=1, batch_first=True)

        if self.require_init:
            self.init_netvlad()

    def init_netvlad(self, use_faiss=True):
        model = self.encoder
        device = torch.device("cuda")
        model.to(device)
        cluster_set = dataset.get_whole_training_set(self.opt, onlyDB=True, forCluster=True)
        nDescriptors = 50000  # the number of descriptors database
        nPerImage = 50  # the number of descriptors each image contributes
        nIm = ceil(nDescriptors / nPerImage)  # the number of sample images

        sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
        data_loader = DataLoader(dataset=cluster_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=True, sampler=sampler)

        initcache = os.path.join(self.opt.runsPath, 'descriptor_center.hdf5')
        if not os.path.exists(initcache):
            with h5py.File(initcache, mode='w') as h5:
                with torch.no_grad():
                    model.eval()
                    print('====> Extracting Descriptors')
                    dbFeat = h5.create_dataset("descriptors", [nDescriptors, self.opt.encoder_dim], dtype=np.float32)
                    print('clustering..')
                    for iteration, (input, indices) in enumerate(data_loader, 1):
                        input = input.to(device)  # ([32, 3, 154, 154])     ([32, 5, 3, 200, 200])
                        image_descriptors = model.encoder(input).view(input.size(0), self.opt.encoder_dim, -1).permute(0, 2, 1)  # ([32, 81, 512])
                        # print(image_descriptors.shape)
                        batchix = (iteration - 1) * self.opt.cacheBatchSize * nPerImage
                        for ix in range(image_descriptors.size(0)):  # 0, 1
                            # sample different location for each image in batch
                            sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)  # choose 50 outof 81
                            startix = batchix + ix * nPerImage  # 0, 50
                            dbFeat[startix:startix + nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                        if iteration % 50 == 0 or len(data_loader) <= 10:
                            print("==> Batch ({}/{})".format(iteration, ceil(nIm / self.opt.cacheBatchSize)), flush=True)
                        del input, image_descriptors

                if use_faiss:
                    # ===============faiss method begin===================
                    print('====> Clustering..')
                    niter = 100
                    kmeans = faiss.Kmeans(self.opt.encoder_dim, self.opt.num_clusters, niter=niter, verbose=False)
                    kmeans.train(dbFeat[...])

                    print('====> Storing centroids', kmeans.centroids.shape)
                    h5.create_dataset('centroids', data=kmeans.centroids)
                    print('====> Done!')
                    # ===============faiss method end===================
                else:
                    # ===============sklearn method begin==================
                    print('====> Clustering..')
                    niter = 100
                    km = KMeans(n_clusters=self.opt.num_clusters, n_init=niter, random_state=156).fit(dbFeat[...])
                    print('====> Storing centroids', km.cluster_centers_.shape)
                    h5.create_dataset('centroids', data=km.cluster_centers_)
                    print('====> Done!')
                    # ===============sklearn method end==================
        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            self.netvlad.init_params(clsts, traindescs)
            del clsts, traindescs

    def forward(self, input):                   # ([96, 3, 200, 200])   ([32, 5, 3, 200, 200])
        batch_size, seq_len, input_c, input_h, input_w = input.shape
        input = input.view(batch_size * seq_len, 3, 200, 200)               # ([32*5, 3, 200, 200])

        feature = self.encoder.encoder(input)   # ([160, 512, 12, 12])
        vlad = self.netvlad(feature)            # ([160, 32768])

        if self.opt.seqLen > 1:
            vlad = vlad.view(batch_size, seq_len, -1)               # ..([32, 5, 32768])
            self.lstm.flatten_parameters()
            vlad, _ = self.lstm(vlad, None)                                         # ..([32, 5, 32768])
            vlad = vlad[:, -1, :]                                                   # ..([32, 32768])
            vlad = F.normalize(vlad, p=2, dim=1)
        return vlad
