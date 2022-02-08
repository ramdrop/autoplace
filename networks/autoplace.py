import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import torchvision.models as models
from torch.utils.data import DataLoader, SubsetRandomSampler
from ipdb import set_trace

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class get_encoder(nn.Module):
    def __init__(self, opt):
        super(get_encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),           # Conv1
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),         # Conv2
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),        # Conv3
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),        # Conv4
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            L2Norm())

    def forward(self, input):
        return self.encoder(input)


class get_model(nn.Module):
    def __init__(self, opt, require_init=True):
        super(get_model, self).__init__()
        self.opt = opt
        self.require_init = require_init
        self.encoder = get_encoder(self.opt)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        if self.require_init:
            pass
        if self.opt.seqLen > 1:
            self.lstm = nn.LSTM(input_size=9216, hidden_size=self.opt.output_dim, num_layers=1, batch_first=True)

    def forward(self, input):                                                  # ([32, 3, 3, 200, 200])
        batch_size, seq_len, input_c, input_h, input_w = input.shape           # (32, 3, 3, 200, 200)
        input = input.view(batch_size * seq_len, input_c, input_h, input_w)    # ([32*3, 3, 200, 200])

        feature = self.encoder(input)                                          # ([96, 256, 12, 12])
        feature = self.pool(feature)                                           # ([96, 256, 6, 6])

        if self.opt.seqLen > 1:
            feature = F.normalize(feature, p=2, dim=1)                                             # ([96, 256, 6, 6])
            _, feat_dim, feat_h, feat_w = feature.shape
            feature = feature.view(batch_size, seq_len, feat_dim * feat_h * feat_w)                # ([32, 3, 9216])
            self.lstm.flatten_parameters()
            feature, _ = self.lstm(feature, None)                                                  # ([32, 3, 9216])
            feature = feature[:, -1, :]                                                            # ([32, 9216])
            feature = F.normalize(feature, p=2, dim=1)                                             # ([32, 9216])
        else:
            feature = feature.view(batch_size, -1)                                                 # ([32, 9216])
            feature = F.normalize(feature, p=2, dim=1)                                             # ([32, 9216])

        return feature
