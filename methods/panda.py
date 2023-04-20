import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .utils.base_method import BaseMethod


class CompactnessLoss(nn.Module):
    def __init__(self, center):
        super(CompactnessLoss, self).__init__()
        self.center = center  # (768, )

    def forward(self, inputs):
        m = inputs.size(1)
        variances = (inputs - self.center).norm(dim=1).pow(2) / m  # (32, )
        return variances.mean()


# contastive svdd
class PANDA(BaseMethod):
    def __init__(self, args, net, optimizer, scheduler):
        super(PANDA, self).__init__(args, net, optimizer, scheduler)

    def get_center(self, train_loader):
        self.net.eval()
        train_feature_space = []
        with torch.no_grad():
            for imgs in train_loader:
                imgs = imgs.to(self.args.device)
                features = self.net.forward_features(imgs)
                train_feature_space.append(features)
            train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
        center = torch.FloatTensor(train_feature_space).mean(dim=0)
        return center

    def forward(self, epoch, inputs, labels, one_epoch_embeds, t, center):
        self.compactness_loss = CompactnessLoss(center.to(self.args.device))

        self.optimizer.zero_grad()
        embeds = self.net.forward_features(inputs)  # (32, 768)
        one_epoch_embeds.append(embeds.detach().cpu())
        loss = self.compactness_loss(embeds)

        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step(epoch)

    def training_epoch(self, density, one_epoch_embeds, task_wise_mean, task_wise_cov, task_wise_train_data_nums, t):
        if self.args.eval.eval_classifier == 'density':
            one_epoch_embeds = torch.cat(one_epoch_embeds)
            one_epoch_embeds = F.normalize(one_epoch_embeds, p=2, dim=1)
            _, _ = density.fit(one_epoch_embeds)
            return density
        else:
            pass


