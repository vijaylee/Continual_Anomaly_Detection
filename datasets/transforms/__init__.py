from torchvision import transforms
from .trans_cutpaste import CutPasteNormal, CutPasteScar, CutPaste3Way
from .maskimg import MaskImg


def aug_transformation(args):
    data_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    if args.dataset.strong_augmentation:
        after_cutpaste_transform = transforms.Compose([
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(*data_norm)
        ])
        if args.dataset.random_aug:
            aug_transformation = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((args.dataset.image_size, args.dataset.image_size)),
                transforms.RandomChoice([
                    CutPasteNormal(transform=after_cutpaste_transform),
                    CutPasteScar(transform=after_cutpaste_transform),
                    MaskImg(args.device, args.dataset.image_size, 0.25, [16, 2], colorJitter=0.1,
                            transform=after_cutpaste_transform)])
            ])
        else:
            aug_transformation = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((args.dataset.image_size, args.dataset.image_size)),
                # MaskImg(args.device, args.dataset.image_size, 0.25, [16, 2], colorJitter=0.1, transform=after_cutpaste_transform)
                CutPasteNormal(transform=after_cutpaste_transform)
            ])
    else:
        aug_transformation = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Resize(args.dataset.image_size),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(*data_norm)
        ])
    return aug_transformation


def no_aug_transformation(args):
    data_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    no_aug_transformation = transforms.Compose([
        transforms.Resize((args.dataset.image_size, args.dataset.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(*data_norm)
    ])
    return no_aug_transformation