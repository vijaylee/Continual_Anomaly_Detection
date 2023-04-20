from utils.buffer import Buffer
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .utils.base_method import BaseMethodwDNE


class DER(BaseMethodwDNE):
    def __init__(self, args, net, optimizer, scheduler):
        super(DER, self).__init__(args, net, optimizer, scheduler)
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
            if self.args.model.with_embeds:
                buf_inputs, buf_embeds = self.buffer.get_data(self.args.train.batch_size)
                _, past_embeds = self.net(buf_inputs)
                loss += self.args.train.alpha * F.mse_loss(past_embeds, buf_embeds)
            else:
                buf_inputs, buf_logits = self.buffer.get_data(self.args.train.batch_size)
                past_logits, _ = self.net(buf_inputs)
                loss += self.args.train.alpha * F.mse_loss(past_logits, buf_logits)

        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step(epoch)

        if self.args.model.with_embeds:
            self.buffer.add_data(examples=inputs[:num], logits=noaug_embeds.data)
        else:
            self.buffer.add_data(examples=inputs[:num], logits=noaug_logits.data)
