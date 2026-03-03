#Global imports
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader


#Local imports
from util.io import load_json
from model.model import AdaSpot
from util.eval import inference
from dataset.frame import ActionSpotInferenceDataset
from util.dataset import load_classes, load_elements
from util.constants import STRIDE, STRIDE_SNB


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--video_path', type=str, help='Path to video file for inference', required=True)
    parser.add_argument('--inference_threshold', type=float, default=0.2, help='Threshold for inference')
    parser.add_argument('--seed', type=int, default=1)
    
    return parser.parse_args()

def dict_to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{
            k: dict_to_namespace(v) for k, v in d.items()
        })
    return d

def update_args(args, config):

    #Update arguments with config file
    args.paths = dict_to_namespace(config['paths'])
    args.data = dict_to_namespace(config['data'])
    args.data.store_dir = args.paths.save_dir + '/store_data'
    args.paths.save_dir = os.path.join(args.paths.save_dir, args.model_name + '-' + str(args.seed)) # allow for multiple seeds
    args.data.frame_dir = args.paths.frame_dir
    args.data.save_dir = args.paths.save_dir
    args.training = dict_to_namespace(config['training'])
    args.model = dict_to_namespace(config['model'])
    args.model.clip_len = args.data.clip_len
    args.model.dataset = args.data.dataset
    args.model.num_classes = args.data.num_classes

    return args


def main(args):

    config_path = args.model_name.split('_')[0] + '/' + args.model_name + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    # Get classes and elements
    classes = load_classes(os.path.join('data', args.data.dataset, 'class.txt'))
    if args.data.dataset == 'f3set':
        elements = load_elements(os.path.join('data', args.data.dataset, 'elements.txt'))
    else:
        elements = None

    # Model
    model = AdaSpot(args_model=args.model, args_training=args.training, classes=classes, elements=elements)

    print('START INFERENCE')
    model.load(torch.load(os.path.join(
        os.getcwd(), args.paths.save_dir, 'checkpoint_best.pt')))
    model.clean_modules() # clean modules to remove unnecessary parameters for inference and speed up evaluation

    stride = STRIDE
    if args.data.dataset == 'soccernetball':
        stride = STRIDE_SNB

    inf_dataset_kwargs = {
        'clip_len': args.data.clip_len, 'overlap_len': args.data.clip_len // 2, 'stride': stride, 'dataset': args.data.dataset, 'size': (args.model.hr_dim[0], args.model.hr_dim[1])
    }

    inference_dataset = ActionSpotInferenceDataset(args.video_path, **inf_dataset_kwargs)

    inference_loader = DataLoader(
        inference_dataset, batch_size = args.training.batch_size,
        shuffle = False, num_workers = args.training.num_workers,
        pin_memory = True, drop_last = False)

    inference(model, inference_loader, classes, threshold = args.inference_threshold)
    
    print('CORRECTLY FINISHED INFERENCE STEP')


if __name__ == '__main__':
    main(get_args())