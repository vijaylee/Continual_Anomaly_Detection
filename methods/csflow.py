import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .utils.base_method import BaseMethod


class CSFlow(BaseMethod):
    def __init__(self, args, net, optimizer, scheduler):
        super(CSFlow, self).__init__(args, net, optimizer, scheduler)


    def forward(self, epoch, inputs, labels, one_epoch_embeds, *args):
        self.optimizer.zero_grad()
        embeds, z, log_jac_det = self.net(inputs)
        # yy, rev_y, zz = self.net.revward(inputs)
        loss = torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - log_jac_det) / z.shape[1]

        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step(epoch)

