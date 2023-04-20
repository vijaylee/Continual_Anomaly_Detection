from utils.buffer import Buffer
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .utils.base_method import BaseMethodwDNE


class FDR(BaseMethodwDNE):
    def __init__(self, args, net, optimizer, scheduler):
        super(FDR, self).__init__(args, net, optimizer, scheduler)
        self.buffer = Buffer(self.args.model.buffer_size, self.args.device)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.current_task = 0
        self.i = 0
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)

    def end_task(self, train_loader):
        self.current_task += 1
        examples_per_task = self.args.model.buffer_size // self.current_task

        if self.current_task > 1:
            buf_x, buf_log, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for ttl in buf_tl.unique():
                idx = (buf_tl == ttl)
                ex, log, tasklab = buf_x[idx], buf_log[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_task)
                self.buffer.add_data(
                    examples=ex[:first],
                    logits=log[:first],
                    task_labels=tasklab[:first])
        counter = 0
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                if isinstance(data, list):
                    inputs = [x.to(self.args.device) for x in data]
                    inputs = torch.cat(inputs, dim=0)
                else:
                    inputs = data.to(self.args.device)

                if self.args.dataset.strong_augmentation:
                    num = int(len(inputs) / 2)
                else:
                    num = int(len(inputs))

                not_aug_inputs = inputs[:num]
                not_aug_logits, not_aug_embeds = self.net(not_aug_inputs)
                if examples_per_task - counter < 0:
                    break
                self.buffer.add_data(examples=not_aug_inputs[:(examples_per_task - counter)],
                                     logits=not_aug_logits.data[:(examples_per_task - counter)],
                                     task_labels=(torch.ones(self.args.train.batch_size) *
                                                  (self.current_task - 1))[:(examples_per_task - counter)])
                counter += self.args.train.batch_size

    def forward(self, epoch, inputs, labels, one_epoch_embeds, t, *args):
        num = self.pre_forward(inputs, t)

        self.optimizer.zero_grad()
        with torch.no_grad():
            noaug_logits, noaug_embeds = self.net(inputs[:num])
            one_epoch_embeds.append(noaug_embeds.cpu())

        logits, embeds = self.net(inputs)
        loss = self.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()

        if not self.buffer.is_empty():
            self.optimizer.zero_grad()
            buf_inputs, buf_logits, _ = self.buffer.get_data(self.args.train.batch_size)
            past_logits, _ = self.net(buf_inputs)
            loss = torch.norm(self.soft(past_logits) - self.soft(buf_logits), 2, 1).mean()
            assert not torch.isnan(loss)
            loss.backward()
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step(epoch)
