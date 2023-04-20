import argparse
import numpy as np
import torch
import random
import yaml
import re

class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value

    def __getattr__(self, attribute):
        raise AttributeError(
            f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")

def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default='./configs/cad.yaml', type=str, help="xxx.yaml")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default="../datasets/mvtec")
    parser.add_argument('--mtd_dir', type=str, default="../datasets/mtd_ano_mask")
    parser.add_argument('--save_checkpoint', type=str2bool, default=False, help='save checkpoint or not.')
    parser.add_argument('--save_path', type=str, default="./checkpoints")
    parser.add_argument('--noise_ratio', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(args)
    with open(args.config_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        print(data)
        for key, value in Namespace(data).__dict__.items():
            vars(args)[key] = value

    set_deterministic(args.seed)

    return args