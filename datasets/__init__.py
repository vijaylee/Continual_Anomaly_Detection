from .seq_mvtec import get_mvtec_dataloaders
from .seq_mtd_mvtec import get_mtd_mvtec_dataloaders, get_joint_mtd_mvtec_dataloaders


def get_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames):
    if args.dataset.name == 'seq-mvtec':
        train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames = get_mvtec_dataloaders(
            args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames)
    elif args.dataset.name == 'seq-mtd-mvtec':
        train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames = get_mtd_mvtec_dataloaders(
            args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames)
    elif args.dataset.name == 'joint-mtd-mvtec':
        train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames = get_joint_mtd_mvtec_dataloaders(
            args, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames)
    return train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, data_train_nums, all_test_filenames