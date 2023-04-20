from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Union
import os
from copy import deepcopy
import matplotlib.ticker as ticker



# @staticmethod
def plot_tsne(labels, embeds, defect_name=None, save_path = None, **kwargs: Dict[str, Any]):
    """t-SNE visualize
    Args:
        labels (Tensor): labels of test and train
        embeds (Tensor): embeds of test and train
        defect_name ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
        save_path ([str], optional): same as <defect_name> in roc_auc. Defaults to None.
        kwargs (Dict[str, Any]): hyper parameters of t-SNE which will change final result
            n_iter (int): > 250, default = 1000
            learning_rate (float): (10-1000), default = 100
            perplexity (float): (5-50), default = 28
            early_exaggeration (float): change it when not converging, default = 12
            angle (float): (0.2-0.8), default = 0.3
            init (str): "random" or "pca", default = "pca"
    """
    tsne = TSNE(
        n_components=2,
        verbose=1,
        n_iter=kwargs.get("n_iter", 1000),
        learning_rate=kwargs.get("learning_rate", 100),
        perplexity=kwargs.get("perplexity", 28),
        early_exaggeration=kwargs.get("early_exaggeration", 12),
        angle=kwargs.get("angle", 0.3),
        init=kwargs.get("init", "pca"),
    )
    embeds, labels = shuffle(embeds, labels)
    tsne_results = tsne.fit_transform(embeds)

    cmap = plt.cm.get_cmap("spring")
    colors = np.vstack((np.array([[0, 1. ,0, 1.]]), cmap([0, 256//3, (2*256)//3])))
    legends = ["good", "anomaly"]
    (_, ax) = plt.subplots(1)
    plt.title(f't-SNE: {defect_name}')
    for label in torch.unique(labels):
        res = tsne_results[torch.where(labels==label)]
        ax.plot(*res.T, marker="*", linestyle="", ms=5, label=legends[label], color=colors[label])
        ax.legend(loc="best")
    plt.xticks([])
    plt.yticks([])

    save_images = save_path if save_path else './tnse_results'
    os.makedirs(save_images, exist_ok=True)
    image_path = os.path.join(save_images, defect_name+'_tsne.pdf') if defect_name else os.path.join(save_images, 'tsne.pdf')
    plt.savefig(image_path)
    plt.close()
    return


def compare_histogram(scores, classes, start=0 ,thresh=2, interval=1, n_bins=64, name=None, save_path=None):
    classes = deepcopy(classes)
    classes[classes > 0] = 1
    scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=120)

    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ticks = np.linspace(start, thresh, interval)
    labels = [str(i) for i in ticks[:-1]] + ['>' + str(thresh)]

    save_images = save_path if save_path else './his_results1'
    os.makedirs(save_images, exist_ok=True)
    image_path = os.path.join(save_images, name + '_his.pdf') if name else os.path.join(save_images, 'his.pdf')

    plt.yticks(rotation=24)
    plt.xlabel(r'$-log(p(z))$', fontsize=10)
    plt.tick_params(labelsize=10)
    plt.autoscale()
    plt.xticks([], [])
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

