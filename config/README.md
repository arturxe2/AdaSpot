# Configuration File Guide

This document describes the structure of a configuration file used to train and evaluate AdaSpot.

Each configuration file is written in JSON format and contains four main sections:

- `paths`
- `data`
- `training`
- `model`

---
## paths

- `frame_dir`: Path to the extracted dataset frames.
- `save_dir`: Directory where checkpoints and logs are saved.

---
## data

- `dataset`: Dataset name. (FineDiving, Tennis, FineGym, F3Set, SoccerNetBall)
- `num_classes`: Number of classes.
- `clip_len`: Frames per clip.
- `epoch_num_frames`: Frames sampled per epoch.
- `mixup`: Enable/disable MixUp augmentation.
- `store_mode`: `store` to generate and save clip partitions (first run), and `load` to load saved partitions (later runs).
 
---
## training

- `batch_size`: Clips per batch.
- `num_epochs`: Total training epochs. 
- `warm_up_epochs`: Learning rate warm-up.
- `start_val_epoch`: When validation starts.
- `learning_rate`: Base learning rate.
- `only_test`: If `true`, evaluation only.
- `criterion`: Validation metric for early stopping.
- `num_workers`: Data loading workers.
- `lowres_loss`: Enable low-resolution loss.
- `highres_loss`: Enable high-resolution loss.

---
## model

- `hr_dim`: High-resolution input size.
- `hr_crop`: High-resolution crop size.
- `lr_dim`: Low-resolution input size.
- `lr_crop`: Low-resolution crop size.
- `roi_size`: RoI size.
- `feature_arch`: Backbone architecture.
- `blocks_temporal`: Enable temporal blocks per stage (GSF).
- `aggregation`: Low-res + High-res aggregation method.
- `temporal_arch`: Temporal modeling module.
- `threshold`: RoI selector threshold parameter.
- `padding`: Backbone convolutions padding type.
