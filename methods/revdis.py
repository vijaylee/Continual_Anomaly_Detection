import torch
import torch.nn as nn
from .utils.base_method import BaseMethod


class RevDis(BaseMethod):
    def __init__(self, args, net, optimizer, scheduler):
        super(RevDis, self).__init__(args, net, optimizer, scheduler)

    def loss_fucntion(self, a, b):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
        return loss

    def forward(self, epoch, inputs, labels, one_epoch_embeds, *args):
        self.optimizer.zero_grad()
        t_outs, outs = self.net(inputs)
        loss = self.loss_fucntion(t_outs, outs)
        loss.backward()
        self.optimizer.step()
