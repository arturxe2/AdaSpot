#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""

#Standard imports
import torch
import numpy as np
import random
import os
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import time

#Local imports
from util.io import load_json, store_json
from dataset.datasets import get_datasets
from dataset.frame import ActionSpotVideoDataset
from util.constants import LABELS_SNB_PATH, STRIDE, STRIDE_SNB, EVAL_SPLITS
from util.eval import evaluate, evaluate_SNB
from model.model import AdaSpot

def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def dict_to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{
            k: dict_to_namespace(v) for k, v in d.items()
        })
    return d

def update_args(args, config):
    #Update config arguments
    config['paths']['save_dir'] = config['paths']['save_dir'] + '/' + args.model_name + '-' + str(args.seed) # allow for multiple seeds

    #Update arguments with config file
    args.paths = dict_to_namespace(config['paths'])
    args.data = dict_to_namespace(config['data'])
    args.data.frame_dir = args.paths.frame_dir
    args.data.save_dir = args.paths.save_dir
    args.data.store_dir = os.path.join(args.paths.save_dir, args.model_name + '-' + str(args.seed))
    args.training = dict_to_namespace(config['training'])
    args.model = dict_to_namespace(config['model'])
    args.model.clip_len = args.data.clip_len
    args.model.dataset = args.data.dataset
    args.model.num_classes = args.data.num_classes

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    
    sched1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                    total_iters=args.warm_up_epochs * num_steps_per_epoch)
    sched2 = CosineAnnealingLR(optimizer,
                    num_steps_per_epoch * cosine_epochs)
    return args.num_epochs, SequentialLR(optimizer, schedulers=[sched1, sched2],
                    milestones=[args.warm_up_epochs * num_steps_per_epoch])

def check_model_dims(data_args):
    """
    Ensure that model input dimensions are in the correct format (list of 2 ints)
    """
    hr_dim = data_args.hr_dim
    lr_dim = data_args.lr_dim
    hr_crop = data_args.hr_crop
    lr_crop = data_args.lr_crop

    for dim in [hr_dim, lr_dim, hr_crop, lr_crop]:
        if isinstance(dim, list):
            if len(dim) != 2:
                raise ValueError('Dimensions must be a list of 2 ints')
            if not all(isinstance(x, int) for x in dim):
                raise ValueError('Dimensions must be a list of 2 ints')
        else:
            raise ValueError('Dimensions must be a list of 2 ints')
    
    return

def main(args):
    
    #Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    config_path = args.model_name.split('_')[0] + '/' + args.model_name + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    # Check labels path for SN-BAS
    if args.data.dataset == 'soccernetball':
        if not os.path.exists(LABELS_SNB_PATH): # check that the path exists
            raise ValueError('Labels path for SN-BAS does not exist. Please update the LABELS_SNB_PATH constant in util/constants.py with the correct path to the labels file for SN-BAS.')

    # Check dimensions
    check_model_dims(args.model)

    # Get classes + train, val, datasets (+ val frames for map evaluation) + elements (for F3Set if necessary)
    classes, train_data, val_data, val_data_frames, elements = get_datasets(args.data, only_test = args.training.only_test)

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.training.batch_size, # It is already random in get one
        pin_memory=True, num_workers=args.training.num_workers,
        prefetch_factor=1, worker_init_fn=worker_init_fn) 
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.training.batch_size,
        pin_memory=True, num_workers=args.training.num_workers,
        prefetch_factor=1, worker_init_fn=worker_init_fn)
                
    # Model
    model = AdaSpot(args_model=args.model, args_training=args.training, classes=classes, elements=elements)

    # Optimizer and scaler
    optimizer, scaler = model.get_optimizer({'lr': args.training.learning_rate})

    # Training loop
    if not args.training.only_test:
        
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args.training, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = 0 if args.training.criterion == 'map' else float('inf')
        epoch = 0

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):  
            
            # Train epoch
            time_train0 = time.time()
            train_loss = model.epoch(
                train_loader, optimizer, scaler, lr_scheduler=lr_scheduler)
            time_train1 = time.time()
            time_train = time_train1 - time_train0
            
            # Val epoch
            time_val0 = time.time()
            val_loss = model.epoch(val_loader)
            time_val1 = time.time()
            time_val = time_val1 - time_val0

            better = False
            val_mAP = 0
            if args.training.criterion == 'loss':
                if val_loss <= best_criterion:
                    best_criterion = val_loss
                    better = True
            elif args.training.criterion == 'map':
                if epoch >= args.training.start_val_epoch:
                    time_map0 = time.time()
                    val_mAP = evaluate(model, val_data_frames, 'VAL', classes,
                                        printed=False, test=False)
                    time_map1 = time.time()
                    time_map = time_map1 - time_map0
                    if val_mAP >= best_criterion:
                        best_criterion = val_mAP
                        better = True
            
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
                epoch, train_loss, val_loss))
            if (args.training.criterion == 'map') & (epoch >= args.training.start_val_epoch):
                print('Val mAP: {:0.5f}'.format(val_mAP))
                if better:
                    print('New best mAP epoch!')
            print('Time train: ' + str(int(time_train // 60)) + 'min ' + str(np.round(time_train % 60, 2)) + 'sec')
            print('Time val: ' + str(int(time_val // 60)) + 'min ' + str(np.round(time_val % 60, 2)) + 'sec')
            if (args.training.criterion == 'map') & (epoch >= args.training.start_val_epoch):
                print('Time map: ' + str(int(time_map // 60)) + 'min ' + str(np.round(time_map % 60, 2)) + 'sec')
            else:
                time_map = 0

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss,
                'val_mAP': val_mAP
            })

            if args.paths.save_dir is not None:

                # Store losses
                os.makedirs(args.paths.save_dir, exist_ok=True)
                store_json(os.path.join(args.paths.save_dir, 'loss.json'), losses,
                            pretty=True)

                # Store model (if better)
                if better:
                    torch.save(
                        model.state_dict(),
                        os.path.join(os.getcwd(), args.paths.save_dir, 'checkpoint_best.pt'))

    print('START INFERENCE')
    # Load best model
    model.load(torch.load(os.path.join(
        os.getcwd(), args.paths.save_dir, 'checkpoint_best.pt')))
    model.clean_modules() # clean modules to remove unnecessary parameters for inference and speed up evaluation

    eval_splits = EVAL_SPLITS

    for split in eval_splits:
        split_path = os.path.join(
            'data', args.data.dataset, '{}.json'.format(split))

        stride = STRIDE
        if args.data.dataset == 'soccernetball':
            stride = STRIDE_SNB

        if os.path.exists(split_path):
            
            val_dataset_kwargs = {
                'classes': classes, 'frame_dir': args.data.frame_dir, 'clip_len': args.data.clip_len, 'dataset': args.data.dataset, 
                'stride': stride, 'overlap_len': args.data.clip_len // 2
                }
            split_data = ActionSpotVideoDataset(split_path, **val_dataset_kwargs)

            pred_file = None
            if args.paths.save_dir is not None:
                pred_file = os.path.join(
                    args.paths.save_dir, 'pred-{}'.format(split))
            
            mAPs, tolerances = evaluate(model, split_data, split.upper(), classes, pred_file, printed = True, 
                test = True)

            if args.data.dataset == 'soccernetball':
                results = evaluate_SNB(LABELS_SNB_PATH, '/'.join(pred_file.split('/')[:-1]) + '/preds', split = split, metric = 'at1', classes = classes)
                
                print('aMAP@1: ', results['a_mAP'] * 100)
                print('Average mAP per class: ')
                print('-----------------------------------')
                for i in range(len(results["a_mAP_per_class"])):
                    print("    " + list(classes.keys())[i] + ": " + str(np.round(results["a_mAP_per_class"][i] * 100, 2)))

                results_2 = evaluate_SNB(LABELS_SNB_PATH, '/'.join(pred_file.split('/')[:-1]) + '/preds', split = split, metric = 'at2', classes = classes)
                print('aMAP@2: ', results_2['a_mAP'] * 100)
                print('Average mAP@2 per class: ')
                print('-----------------------------------')
                for i in range(len(results_2["a_mAP_per_class"])):
                    print("    " + list(classes.keys())[i] + ": " + str(np.round(results_2["a_mAP_per_class"][i] * 100, 2)))

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')




if __name__ == '__main__':
    main(get_args())