from tqdm import tqdm
import torch
from torch import optim
import os
import numpy as np
from eval import eval_model
from argument import get_args
from models import get_net_optimizer_scheduler
from methods import get_model
from datasets import get_dataloaders
from utils.density import GaussianDensityTorch



def get_inputs_labels(data):
    if isinstance(data, list):
        inputs = [x.to(args.device) for x in data]
        labels = torch.arange(len(inputs), device=args.device)
        labels = labels.repeat_interleave(inputs[0].size(0))
        inputs = torch.cat(inputs, dim=0)
    else:
        inputs = data.to(args.device)
        labels = torch.zeros(inputs.size(0), device=args.device).long()
    return inputs, labels


def main(args):
    net, optimizer, scheduler = get_net_optimizer_scheduler(args)
    density = GaussianDensityTorch()
    net.to(args.device)

    model = get_model(args, net, optimizer, scheduler)

    dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames = [], [], [], []
    task_wise_mean, task_wise_cov, task_wise_train_data_nums = [], [], []
    for t in range(args.dataset.n_tasks):
        print('---' * 10, f'Task:{t}', '---' * 10)
        train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames = get_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames)
        task_wise_train_data_nums.append(data_train_nums)

        extra_para = None
        if args.model.method == 'panda':
            extra_para = model.get_center(train_dataloader)

        net.train()
        for epoch in tqdm(range(args.train.num_epochs)):
            one_epoch_embeds = []
            if args.model.method == 'upper':
                for dataloader_train in dataloaders_train:
                    for batch_idx, (data) in enumerate(dataloader_train):
                        inputs, labels = get_inputs_labels(data)
                        model(epoch, inputs, labels, one_epoch_embeds, t, extra_para)
            else:
                for batch_idx, (data) in enumerate(train_dataloader):
                    inputs, labels = get_inputs_labels(data)
                    model(epoch, inputs, labels, one_epoch_embeds, t, extra_para)

            if args.train.test_epochs > 0 and (epoch+1) % args.train.test_epochs == 0:
                net.eval()
                density = model.training_epoch(density, one_epoch_embeds, task_wise_mean, task_wise_cov, task_wise_train_data_nums, t)
                eval_model(args, epoch, dataloaders_test, learned_tasks, net, density)

        if hasattr(model, 'end_task'):
            model.end_task(train_dataloader)

    if args.save_checkpoint:
        torch.save(net,  f'{args.save_path}/net.pth')
        torch.save(density, f'{args.save_path}/density.pth')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = get_args()
    main(args)
