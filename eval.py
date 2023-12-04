from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from argument import get_args
from datasets import get_dataloaders
from utils.visualization import plot_tsne, compare_histogram, cal_anomaly_map



def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def csflow_eval(args, epoch, dataloaders_test, learned_tasks, net):
    all_roc_auc = []
    eval_task_wise_scores, eval_task_wise_labels = [], []
    task_num = 0
    for idx, (dataloader_test, learned_task) in enumerate(zip(dataloaders_test, learned_tasks)):
        test_z, test_labels = list(), list()

        with torch.no_grad():
            for i, data in enumerate(dataloader_test):
                inputs, labels = data
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                _, z, jac = net(inputs)
                z = t2np(z[..., None])
                score = np.mean(z ** 2, axis=(1, 2))
                test_z.append(score)
                test_labels.append(t2np(labels))

        test_labels = np.concatenate(test_labels)
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        anomaly_score = np.concatenate(test_z, axis=0)
        roc_auc = roc_auc_score(is_anomaly, anomaly_score)
        all_roc_auc.append(roc_auc * len(learned_task))
        task_num += len(learned_task)
        print('data_type:', learned_task, 'auc:', roc_auc, '**' * 11)

        eval_task_wise_scores.append(anomaly_score)
        eval_task_wise_scores_np = np.concatenate(eval_task_wise_scores)
        eval_task_wise_labels.append(is_anomaly)
        eval_task_wise_labels_np = np.concatenate(eval_task_wise_labels)
    print('mean_auc:', np.sum(all_roc_auc) / task_num, '**' * 11)

    if args.eval.visualization:
        name = f'{args.model.method}_task{len(learned_tasks)}_epoch{epoch}'
        his_save_path = f'./his_results/{args.model.method}{args.model.name}_{args.train.num_epochs}_epochs_seed{args.seed}'
        compare_histogram(np.array(eval_task_wise_scores_np), np.array(eval_task_wise_labels_np), start=0, thresh=5,
                          interval=1, name=name, save_path=his_save_path)


def revdis_eval(args, epoch, dataloaders_test, learned_tasks, net):
    all_roc_auc = []
    eval_task_wise_scores, eval_task_wise_labels = [], []
    task_num = 0
    for idx, (dataloader_test, learned_task) in enumerate(zip(dataloaders_test, learned_tasks)):
        gt_list_sp, pr_list_sp = [], []
        with torch.no_grad():
            for img, gt, label in dataloader_test:
                img = img.to(args.device)
                inputs = net.encoder(img)
                outputs = net.decoder(net.bn(inputs))
                anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
                pr_list_sp.append(np.max(anomaly_map))

        roc_auc = roc_auc_score(gt_list_sp, pr_list_sp)
        all_roc_auc.append(roc_auc * len(learned_task))
        task_num += len(learned_task)
        print('data_type:', learned_task, 'auc:', roc_auc, '**' * 11)

        eval_task_wise_scores.append(pr_list_sp)
        eval_task_wise_scores_np = np.concatenate(eval_task_wise_scores)
        eval_task_wise_labels.append(gt_list_sp)
        eval_task_wise_labels_np = np.concatenate(eval_task_wise_labels)
    print('mean_auc:', np.sum(all_roc_auc) / task_num, '**' * 11)

    if args.eval.visualization:
        name = f'{args.model.method}_task{len(learned_tasks)}_epoch{epoch}'
        his_save_path = f'./his_results/{args.model.method}{args.model.name}_{args.train.num_epochs}_epochs_seed{args.seed}'
        compare_histogram(np.array(eval_task_wise_scores_np), np.array(eval_task_wise_labels_np), thresh=2, interval=1,
                          name=name, save_path=his_save_path)



def eval_model(args, epoch, dataloaders_test, learned_tasks, net, density):
    if args.model.method == 'csflow':
        csflow_eval(args, epoch, dataloaders_test, learned_tasks, net)
    elif args.model.method == 'revdis':
        revdis_eval(args, epoch, dataloaders_test, learned_tasks, net)
    else:
        all_roc_auc, all_embeds, all_labels = [], [], []
        task_num = 0
        for idx, (dataloader_test,  learned_task) in enumerate(zip(dataloaders_test, learned_tasks)):
            labels, embeds, logits = [], [], []
            with torch.no_grad():
                for x, label in dataloader_test:
                    logit, embed = net(x.to(args.device))
                    _, logit = torch.max(logit, 1)
                    logits.append(logit.cpu())
                    embeds.append(embed.cpu())
                    labels.append(label.cpu())
            labels, embeds, logits = torch.cat(labels), torch.cat(embeds), torch.cat(logits)
            # norm embeds
            if args.eval.eval_classifier == 'density':
                embeds = F.normalize(embeds, p=2, dim=1)  # embeds.shape=(2*bs, emd_dim)
                distances = density.predict(embeds)  # distances.shape=(2*bs)
                fpr, tpr, _ = roc_curve(labels, distances)
            elif args.eval.eval_classifier == 'head':
                fpr, tpr, _ = roc_curve(labels, logits)
            roc_auc = auc(fpr, tpr)
            all_roc_auc.append(roc_auc * len(learned_task))
            task_num += len(learned_task)
            all_embeds.append(embeds)
            all_labels.append(labels)
            print('data_type:', learned_task[:], 'auc:', roc_auc, '**' * 11)

            if args.eval.visualization:
                name = f'{args.model.method}_task{len(learned_tasks)}_{learned_task[0]}_epoch{epoch}'
                his_save_path = f'./his_results/{args.model.method}{args.model.name}_{args.train.num_epochs}e_order{args.data_order}_seed{args.seed}'
                tnse_save_path = f'./tsne_results/{args.model.method}{args.model.name}_{args.train.num_epochs}e_order{args.data_order}_seed{args.seed}'
                plot_tsne(labels, np.array(embeds), defect_name=name, save_path=tnse_save_path)
                # These parameters can be modified based on the visualization effect
                start, thresh, interval = 0, 120, 1
                compare_histogram(np.array(distances), labels, start=start,
                                  thresh=thresh, interval=interval,
                                  name=name, save_path=his_save_path)

        print('mean_auc:', np.sum(all_roc_auc) / task_num, '**' * 11)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    args = get_args()
    dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames = [], [], [], []
    for t in range(args.dataset.n_tasks):
        train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames = get_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames)

    epoch = args.train.num_epochs
    net, density = torch.load(f'{args.save_path}/net.pth'), torch.load(f'{args.save_path}/density.pth')
    eval_model(args, epoch, dataloaders_test, learned_tasks, net, density)
