import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.rd_resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from utils.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
import torch.hub



class NetRevDis(nn.Module):
    def __init__(self, args):
        super(NetRevDis, self).__init__()
        torch.hub.set_dir('./checkpoints')
        self.args = args
        self.encoder, self.bn = wide_resnet50_2(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = de_wide_resnet50_2(pretrained=False)

    def forward(self, imgs):
        inputs = self.encoder(imgs)
        outputs = self.decoder(self.bn(inputs))
        return inputs, outputs