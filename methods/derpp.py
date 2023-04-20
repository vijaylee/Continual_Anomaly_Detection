from utils.buffer import Buffer
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .utils.base_method import BaseMethodwDNE


class DERpp(BaseMethodwDNE):
    def __init__(self, args, net, optimizer, scheduler):
        super(DERpp, self).__init__(args, net, optimizer, scheduler)
        self.buffer = Buffer(self.args.model.buffer_size, self.args.device)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, epoch, inputs, labels, one_epoch_embeds, t, *args):
        num = self.pre_forward(inputs, t)

        self.optimizer.zero_grad()
        with torch.no_grad():
            noaug_logits, noaug_embeds = self.net(inputs[:num])
            one_epoch_embeds.append(noaug_embeds.cpu())

        logits, embeds = self.net(inputs)
        loss = self.cross_entropy(logits, labels)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(self.args.train.batch_size)
            past_logits, _ = self.net(buf_inputs)
            loss += (self.args.train.alpha * F.mse_loss(past_logits, buf_logits)
                     + self.args.train.beta * self.cross_entropy(past_logits, buf_labels))

        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step(epoch)

        self.buffer.add_data(examples=inputs[:num], labels=labels[:num], logits=noaug_logits.data)


