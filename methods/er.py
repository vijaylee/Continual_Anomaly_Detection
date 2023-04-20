from utils.buffer import Buffer
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .utils.base_method import BaseMethodwDNE


class ER(BaseMethodwDNE):
    def __init__(self, args, net, optimizer, scheduler):
        super(ER, self).__init__(args, net, optimizer, scheduler)
        self.buffer = Buffer(self.args.model.buffer_size, self.args.device)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, epoch, inputs, labels, one_epoch_embeds, t, *args):
        num = self.pre_forward(inputs, t)

        self.optimizer.zero_grad()
        with torch.no_grad():
            noaug_logits, noaug_embeds = self.net(inputs[:num])
            one_epoch_embeds.append(noaug_embeds.cpu())

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.train.batch_size)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        logits, _ = self.net(inputs)
        loss = self.cross_entropy(logits, labels)

        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step(epoch)

        self.buffer.add_data(examples=inputs[:num], labels=labels[:num])

