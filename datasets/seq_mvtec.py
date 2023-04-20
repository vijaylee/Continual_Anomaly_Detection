from torchvision import transforms
from .transforms import aug_transformation, no_aug_transformation
from .mvtec_dataset import MVTecAD
from .revdis_mvtec_dataset import RevDisTestMVTecDataset
from torch.utils.data import DataLoader
from .utils import get_mvtec_classes


def get_mvtec_dataloaders(args, t, dataloaders_train, dataloaders_test, learned_tasks, all_test_filenames):
    mvtec_classes = get_mvtec_classes(args)

    N_CLASSES_PER_TASK = args.dataset.n_classes_per_task
    if args.dataset.data_incre_setting == 'one':
        # N_CLASSES_PER_TASK = 1
        if t == 0:
            task_mvtec_classes = mvtec_classes[: 10]
        else:
            i = 10 + (t - 1) * N_CLASSES_PER_TASK
            task_mvtec_classes = mvtec_classes[i: i + N_CLASSES_PER_TASK]
    else:
        # N_CLASSES_PER_TASK = 3
        i = t * N_CLASSES_PER_TASK
        task_mvtec_classes = mvtec_classes[i: i + N_CLASSES_PER_TASK]
    learned_tasks.append(task_mvtec_classes)

    train_transform = aug_transformation(args)
    test_transform = no_aug_transformation(args)

    if args.model.method == 'revdis':
        train_data = MVTecAD(args.data_dir, task_mvtec_classes, transform=test_transform, size=args.dataset.image_size)
        test_data = RevDisTestMVTecDataset(args.data_dir, task_mvtec_classes, size=args.dataset.image_size)
        all_test_filenames.append(test_data.img_paths)
    else:
        train_data = MVTecAD(args.data_dir, task_mvtec_classes, transform=train_transform, size=args.dataset.image_size)
        test_data = MVTecAD(args.data_dir, task_mvtec_classes, args.dataset.image_size, transform=test_transform, mode="test")
        all_test_filenames.append(test_data.all_image_names)

    train_dataloader = DataLoader(train_data, batch_size=args.train.batch_size, shuffle=True, num_workers=args.dataset.num_workers)
    dataloaders_train.append(train_dataloader)
    dataloader_test = DataLoader(test_data, batch_size=args.eval.batch_size, shuffle=False, num_workers=args.dataset.num_workers)
    dataloaders_test.append(dataloader_test)
    print('class name:', task_mvtec_classes, 'number of training sets:', len(train_data),
          'number of testing sets:', len(test_data))

    return train_dataloader, dataloaders_train, dataloaders_test, learned_tasks, len(train_data), all_test_filenames