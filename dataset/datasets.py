# Standard imports
import os

# Local imports
from util.dataset import load_classes, load_elements
from util.constants import STRIDE, STRIDE_SNB, OVERLAP
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset


def get_datasets(args, only_test = False):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))
    elements = None
    if args.dataset == 'f3set':
        elements = load_elements(os.path.join('data', args.dataset, 'elements.txt'))

    if only_test:
        return classes, None, None, None, elements

    dataset_len = args.epoch_num_frames // args.clip_len
    if args.dataset == 'soccernetball':
        stride = STRIDE_SNB
    else:
        stride = STRIDE
    overlap = OVERLAP

    dataset_kwargs = {
        'classes': classes, 'frame_dir': args.frame_dir, 'store_dir': args.store_dir, 'store_mode': args.store_mode,
        'clip_len': args.clip_len, 'dataset_len': dataset_len, 'dataset': args.dataset, 'stride': stride, 
        'overlap': overlap, 'mixup': args.mixup, 'elements': elements
    }

    print('Dataset size:', dataset_len)

    train_data = ActionSpotDataset(
        os.path.join('data', args.dataset, 'train.json'), **dataset_kwargs)
    train_data.print_info()
        
    dataset_kwargs['mixup'] = False # Disable mixup for validation

    val_data = ActionSpotDataset(
        os.path.join('data', args.dataset, 'val.json'), **dataset_kwargs)
    val_data.print_info()

    val_dataset_kwargs = {
        'classes': classes, 'frame_dir': args.frame_dir, 'clip_len': args.clip_len, 'dataset': args.dataset, 
        'stride': stride, 'overlap_len': 0
    }
    val_data_frames = ActionSpotVideoDataset(
        os.path.join('data', args.dataset, 'val.json'), **val_dataset_kwargs)
    val_data_frames.print_info()
        
    return classes, train_data, val_data, val_data_frames, elements