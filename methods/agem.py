from utils.buffer import Buffer
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .utils.base_method import BaseMethodwDNE
try:
    import quadprog
except:
    print('Warning: GEM and A-GEM cannot be used on Windows (quadprog required)')



def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1

def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGEM(BaseMethodwDNE):
    def __init__(self, args, net, optimizer, scheduler):
        super(AGEM, self).__init__(args, net, optimizer, scheduler)
        self.buffer = Buffer(self.args.model.buffer_size, self.args.device)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.current_task = 0
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.args.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.args.device)

    def end_task(self, train_loader):
        data = next(iter(train_loader))
        cur_x = [x for x in data]
        cur_y = torch.arange(len(cur_x))
        cur_y = cur_y.repeat_interleave(cur_x[0].size(0))
        cur_x = torch.cat(cur_x, dim=0)
        self.buffer.add_data(
            examples=cur_x.to(self.args.device),
            labels=cur_y.to(self.args.device)
        )

    def forward(self, epoch, inputs, labels, one_epoch_embeds, t, *args):
        num = self.pre_forward(inputs, t)

        with torch.no_grad():
            noaug_logits, noaug_embeds = self.net(inputs[:num])
            one_epoch_embeds.append(noaug_embeds.cpu())

        self.optimizer.zero_grad()
        logits, embeds = self.net(inputs)
        loss = self.cross_entropy(logits, labels)
        loss.backward()

        if not self.buffer.is_empty():
            store_grad(self.parameters, self.grad_xy, self.grad_dims)

            buf_inputs, buf_labels = self.buffer.get_data(self.args.train.batch_size)
            self.net.zero_grad()
            buf_outputs, _ = self.net(buf_inputs)
            penalty = self.cross_entropy(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.parameters, self.grad_xy, self.grad_dims)

        self.optimizer.step()
