import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils.base_method import BaseMethod



class CutPaste(BaseMethod):
    def __init__(self, args, net, optimizer, scheduler):
        super(CutPaste, self).__init__(args, net, optimizer, scheduler)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, epoch, inputs, labels, one_epoch_embeds, *args):
        if self.args.dataset.strong_augmentation:
            half_num = int(len(inputs) / 2)
            no_strongaug_inputs = inputs[:half_num]
        else:
            no_strongaug_inputs = inputs

        self.optimizer.zero_grad()
        with torch.no_grad():
            noaug_embeds = self.net.forward_features(no_strongaug_inputs)
            one_epoch_embeds.append(noaug_embeds.cpu())
        out, _ = self.net(inputs)
        loss = self.cross_entropy(out, labels)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step(epoch)

    def training_epoch(self, density, one_epoch_embeds, *args):
        if self.args.eval.eval_classifier == 'density':
            one_epoch_embeds = torch.cat(one_epoch_embeds)
            one_epoch_embeds = F.normalize(one_epoch_embeds, p=2, dim=1)
            _, _ = density.fit(one_epoch_embeds)
            return density
        else:
            pass



