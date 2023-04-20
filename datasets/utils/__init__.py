

def get_mvtec_classes(args):
    if args.dataset.dataset_order == 1:
        mvtec_classes = ['leather', 'bottle', 'metal_nut',
                         'grid', 'screw', 'zipper',
                         'tile', 'hazelnut', 'toothbrush',
                         'wood', 'transistor', 'pill',
                         'carpet', 'capsule', 'cable']
    elif args.dataset.dataset_order == 2:
        mvtec_classes = ['wood', 'transistor', 'pill',
                         'tile', 'hazelnut', 'toothbrush',
                         'leather', 'bottle', 'metal_nut',
                         'carpet', 'capsule', 'cable',
                         'grid', 'screw', 'zipper']
    elif args.dataset.dataset_order == 3:
        mvtec_classes = ['leather', 'grid', 'tile',
                         'bottle', 'toothbrush', 'capsule',
                         'screw', 'pill', 'zipper',
                         'cable', 'metal_nut', 'hazelnut',
                         'wood', 'carpet', 'transistor']
    return mvtec_classes
