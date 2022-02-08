import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torchvision.models as models
import nuscene as dataset
import os
import math
from math import ceil
import faiss
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.cluster import KMeans
import h5py
import antialiased_cnns
import re


class MaxPool2DWrapper(nn.Module):
    def forward(self, input):
        W = input.shape[-1]
        return F.max_pool2d(input, kernel_size=(W, 1), stride=1)       # 'along amimuth axis'


class CircularPadding(nn.Module):
    def forward(self, input):
        B, C, H, W = input.shape
        k = 3
        delta = int((k - 1) / 2)
        left = input[:, :, :, 0:delta]
        right = input[:, :, :, W - delta:]
        # print('input:{}'.format(input.shape))
        # print('left:{}, right:{}'.format(left.shape, right.shape))
        new_input_data = torch.cat((right, input, left), dim=-1)
        new_input_data.shape
        # print('new input:{}'.format(new_input_data.shape))

        new_W = new_input_data.shape[-1]
        up = new_input_data[:, :, H - delta:, :]
        down = new_input_data[:, :, 0:delta, :]
        # print('up:{}, down:{}'.format(up.shape, down.shape))

        if new_W % 2 == 0:
            up_left = up[:, :, :, :int(new_W / 2)]
            up_right = up[:, :, :, new_W - int(new_W / 2):]
            down_left = down[:, :, :, :int(new_W / 2)]
            down_right = down[:, :, :, new_W - int(new_W / 2):]
            # print('up left:{}, up right:{}'.format(up_left.shape, up_right.shape))
            # print('down left:{}, down right:{}'.format(down_left.shape, down_right.shape))

        else:
            up_left = up[:, :, :, :int(new_W / 2)]
            up_right = up[:, :, :, new_W - int(new_W / 2) - 1:]
            down_left = down[:, :, :, :int(new_W / 2)]
            down_right = down[:, :, :, new_W - int(new_W / 2) - 1:]
            # print('up left:{}, up right:{}'.format(up_left.shape, up_right.shape))
            # print('down left:{}, down right:{}'.format(down_left.shape, down_right.shape))

        to_up = torch.cat((down_right, down_left), axis=-1)
        to_down = torch.cat((up_right, up_left), axis=-1)
        # print('to_up:{}, to_down:{}'.format(to_up.shape, to_down.shape))

        final_input_data = torch.cat((to_down, new_input_data, to_up), axis=-2)
        # print('final_input_data:{}'.format(final_input_data.shape))

        # print(input_data)
        # print(final_input_data)
        return final_input_data


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
            # x_flatten.unsqueeze(0).permute(1, 0, 2, 3) (32, 1, 512, 81)
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[C:C + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C:C + 1, :].unsqueeze(2)
            vlad[:, C:C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        # vlad = self.bn(self.linear(vlad))
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

    # === Kidnapped Radar
    # https://github.com/adobe/antialiased-cnns
    kid_layers = list()
    output_dim = 0
    for layer in layers:
        matchObj = re.search(r'Max', str(layer))
        if matchObj is not None:
            kid_layers.append(nn.MaxPool2d(kernel_size=2, stride=1))    # conv1
            kid_layers.append(antialiased_cnns.BlurPool(output_dim, stride=2, filt_size=7))    # conv1
            # The original BlurPool uses a LowPassFilter, I modify it to use a GaussianFilter with filter_size=7, standard_deviation=1, the GaussianFilter array is:
            # a = np.array([0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006])
        else:
            kid_layers.append(CircularPadding())    # conv1
            kid_layers.append(layer)    # conv1

            matchObj = re.search(r'Conv', str(layer))
            if matchObj is not None:
                matchObj = re.search(r'[1-9]*, kernel', str(layer))
                if matchObj is not None:
                    a = matchObj.regs
                    startIndex = a[0][0]
                    endIndex = a[0][1]
                    output_dim = int(str(layer)[startIndex:endIndex].split(',')[0])
    # W = 12
    # H = 12
    # kid_layers.append(nn.MaxPool2d(kernel_size=(1, H), stride=1))    # do not use this one. the result shows (1,H) leads to worse performance than (W,1) when I checked if I was wrong with the pool axis
    # kid_layers.append(nn.MaxPool2d(kernel_size=(W, 1), stride=1))    # 'along amimuth axis'
    kid_layers.append(MaxPool2DWrapper())       # 'along amimuth axis'
    kid_layers.append(L2Norm())
    # ====================

    encoder = nn.Sequential(*kid_layers)
    model = nn.Module()
    model.add_module('encoder', encoder)
    return model


class get_model(nn.Module):
    def __init__(self, opt, require_init=True):
        super(get_model, self).__init__()
        self.opt = opt
        self.require_init = require_init
        self.encoder = get_encoder(self.opt)
        self.pool = NetVLAD(opt, num_clusters=opt.num_clusters, dim=opt.encoder_dim)    # must be 64, 512
        if self.require_init:
            self.init_netvlad()

    def init_netvlad(self, use_faiss=True):
        model = self.encoder
        device = torch.device("cuda")
        print('network.py device: {} {}'.format(device, torch.cuda.current_device()))
        model.to(device)
        cluster_set = dataset.get_whole_training_set(self.opt, onlyDB=True, forCluster=True)
        nDescriptors = 24000  # the number of descriptors database
        nPerImage = 6  # 50, 20             # the number of descriptors each image contributes
        nIm = ceil(nDescriptors / nPerImage)  # the number of sample images

        sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
        data_loader = DataLoader(dataset=cluster_set, num_workers=self.opt.threads, batch_size=self.opt.cacheBatchSize, shuffle=False, pin_memory=True, sampler=sampler)

        initcache = os.path.join(self.opt.runsPath, 'descriptor_center.hdf5')
        if not os.path.exists(initcache):
            with h5py.File(initcache, mode='w') as h5:
                with torch.no_grad():
                    model.eval()
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
                    niter = 100
                    kmeans = faiss.Kmeans(self.opt.encoder_dim, self.opt.num_clusters, niter=niter, verbose=False)
                    kmeans.train(dbFeat[...])

                    h5.create_dataset('centroids', data=kmeans.centroids)
                    print('stored centroids', kmeans.centroids.shape)

                    # ===============faiss method end===================
                else:
                    # ===============sklearn method begin==================
                    niter = 100
                    km = KMeans(n_clusters=self.opt.num_clusters, n_init=niter, random_state=156).fit(dbFeat[...])
                    print('stored centroids', km.cluster_centers_.shape)
                    h5.create_dataset('centroids', data=km.cluster_centers_)
                    # ===============sklearn method end==================
        with h5py.File(initcache, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            self.pool.init_params(clsts, traindescs)
            del clsts, traindescs

    def forward(self, input):                   # ([96, 3, 200, 200])   ([32, 5, 3, 200, 200])
        input_shape = input.shape
        input = input.view(input_shape[0] * input_shape[1], 3, 200, 200)    # ([32*3, 3, 200, 200])

        feature = self.encoder.encoder(input)   # ([96, 512, 12, 12])
        feature_shape = feature.shape
        feature = feature.view(input_shape[0], input_shape[1], feature_shape[1], feature_shape[2], feature_shape[3])    # ([32, 3, 512, 12, 12])
        feature = feature.transpose(1, 2)   # ([32, 512, 3, 12, 12])
        feature = feature.contiguous()
        feature = feature.view(input_shape[0], feature_shape[1], input_shape[1] * feature_shape[2], feature_shape[3])   # ([32, 512, 36, 12])
        vlad = self.pool(feature)            # ([96, output_dim])

        return vlad
