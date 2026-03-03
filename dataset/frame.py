# Global imports
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import pickle
import random
import torchvision
import torch
import numpy as np
import math
import copy

# Local imports
from util.constants import DEFAULT_PAD_LEN, FPS_SNB, LABELS_SNB_PATH, F3SET_ELEMENTS
from util.io import load_json

class ActionSpotDataset(Dataset):

    def __init__(
            self,
            label_file,                 # path to label json
            classes,                    # dict of class names to idx
            frame_dir,                  # path to frames
            store_dir,                  # path to store files (with frames path and labels per clip)
            store_mode,                 # 'store' or 'load'
            clip_len,                   # Number of frames per clip
            dataset_len,                # Number of clips
            dataset = 'finediving',     # Dataset name
            stride=1,                   # Downsample frame rate
            overlap=0.9,                  # Overlap between clips (in proportion to clip_len)
            mixup=False,                # Mixup usage
            elements = None,            # For F3SET, elements dictionary
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
    ):

        # Initialize variables
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._split = label_file.split('/')[-1].split('.')[0]
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._dataset = dataset
        self._store_dir = store_dir
        self._store_mode = store_mode
        assert store_mode in ['store', 'load']
        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        #    Convert overlap proportion to number of frames
        if overlap != 1:
            self._overlap = int((1-overlap) * clip_len)
        else:
            self._overlap = 1
        assert overlap >= 0 and overlap <= 1
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0
        self.elements = elements
        self._mixup = mixup        

        #Frame reader class
        self._frame_reader = FrameReader(frame_dir, dataset)

        #Store or load clips
        if self._store_mode == 'store':
            self._store_clips()
        elif self._store_mode == 'load':
            self._load_clips()

        self._total_len = len(self._frame_paths)

    def _store_clips(self):
        
        #Initialize frame paths list
        self._frame_paths = []
        self._labels_store = []
        if self._dataset == 'f3set':
            self._labels_elements_store = []
        
        # Iterate over labels
        for video in tqdm(self._labels):

            video_len = int(video['num_frames'])


            if self._dataset == 'soccernetball':
                labels_file = load_json(os.path.join(LABELS_SNB_PATH, video['video'] + '/Labels-ball.json'))['annotations']
            else:
                labels_file = video['events']

            for base_idx in range(-self._pad_len * self._stride, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride), self._overlap):

                if self._dataset == 'finegym':
                    frames_paths = self._frame_reader.load_paths(video['video'], base_idx, base_idx + self._clip_len * self._stride, stride=self._stride, 
                                        source_info = video['_source_info'])
                else:
                    frames_paths = self._frame_reader.load_paths(video['video'], base_idx, base_idx + self._clip_len * self._stride, stride=self._stride)

                labels = []
                if self._dataset == 'f3set':
                    labels_elements = []
                
                # Iterate over label events
                for event in labels_file:
                    
                    if self._dataset == 'soccernetball':
                        event_frame = int(int(event['position']) / 1000 * FPS_SNB) #miliseconds to frames
                    else:
                        event_frame = event['frame']
                    label_idx = (event_frame - base_idx) // self._stride

                    if (label_idx >= 0 and label_idx < self._clip_len):
                        if event['label'] not in self._class_dict: #Some labels in SoccerNetBall are not considered
                            continue
                        label = self._class_dict[event['label']]
                        if self._dataset == 'f3set':
                            label_name = event['label'].split('_')
                            label_elements = []
                            for i in range(len(label_name)):
                                label_elements.append(self.elements[i][label_name[i]])
                        for i in range(max(0, label_idx), min(self._clip_len, label_idx + 1)):
                            labels.append({'label': label, 'label_idx': i})
                            if self._dataset == 'f3set':
                                labels_elements.append({'label_elements': label_elements, 'label_idx': i})

                # Keep only clips with existing frames
                if frames_paths[1] != -1:
                        
                    self._frame_paths.append(frames_paths)
                    self._labels_store.append(labels)
                    if self._dataset == 'f3set':
                        self._labels_elements_store.append(labels_elements)
                            
        #Save to store
        store_path = os.path.join(self._store_dir, self._dataset, 'LEN' + str(self._clip_len) + 'SPLIT' + self._split)

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        if self._dataset == 'f3set':
            with open(store_path + '/labelsE.pkl', 'wb') as f:
                pickle.dump(self._labels_elements_store, f)
        print('Stored clips to ' + store_path)
        return
    
    def _load_clips(self):
        store_path = os.path.join(self._store_dir, self._dataset, 'LEN' + str(self._clip_len) + 'SPLIT' + self._split)
        
        if not os.path.exists(store_path):
            raise ValueError('Store path does not exist. Please run with store mode first to store the clips before loading.')
        
        with open(store_path + '/frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + '/labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)
        if self._dataset == 'f3set':
            with open(store_path + '/labelsE.pkl', 'rb') as f:
                self._labels_elements_store = pickle.load(f)
        print('Loaded clips from ' + store_path)
        return

    def _get_one(self):
        #Get random index
        idx = random.randint(0, self._total_len - 1)

        #Get frame_path and labels dict
        frames_path = self._frame_paths[idx]
        dict_label = self._labels_store[idx]
        if self._dataset == 'f3set':
            dict_labelE = self._labels_elements_store[idx]        

        #Load frames
        frames = self._frame_reader.load_frames(frames_path, pad=True, stride=self._stride)

        #Process labels
        labels = np.zeros(self._clip_len, np.int64)
        for label in dict_label:
            labels[label['label_idx']] = label['label']

        #Process F3SET elements labels
        if self._dataset == 'f3set':
            labelsE = np.zeros((len(F3SET_ELEMENTS) + 1, self._clip_len), np.int64)
            labelsE[1:] = -1  # Initialize to -1 (except binary background event in first)
            for label in dict_labelE:
                labelsE[0][label['label_idx']] = 1  # Background event
                for i in range(len(label['label_elements'])):
                    labelsE[i+1][label['label_idx']] = label['label_elements'][i]

        output = {}
        output['frame'] = frames
        output['contains_event'] = int(np.sum(labels) > 0)
        output['label'] = labels
        if self._dataset == 'f3set':
            output['labelE'] = labelsE

        return output

    def __getitem__(self, unused):
        ret = self._get_one()
        
        if self._mixup:
            mix = self._get_one()    # Sample another clip
            
            ret['frame2'] = mix['frame']
            ret['contains_event2'] = mix['contains_event']
            ret['label2'] = mix['label']
            if self._dataset == 'f3set':
                ret['labelE2'] = mix['labelE']

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)

class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            label_file,                 # path to label json
            classes,                    # dict of class names to idx
            frame_dir,                  # path to frames
            clip_len,                   # Number of frames per clip
            dataset = 'finediving',     # Dataset name
            stride=1,                   # Downsample frame rate
            overlap_len=0,              # Overlap between clips (in number of frames)
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
    ):      

        # Initialize variables
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._clip_len = clip_len
        self._dataset = dataset
        self._stride = stride
        
        self._frame_reader = FrameReaderVideo(frame_dir, dataset)

        self._clips = []
        
        # Iterate over labels
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                if self._dataset == 'finegym':
                    self._clips.append((l['video'], i, l['_source_info']))
                else:
                    self._clips.append((l['video'], i))
            assert has_clip, l
        
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        if self._dataset == 'finegym':
            video_name, start, source_info = self._clips[idx]
        else:
            video_name, start = self._clips[idx]

        if self._dataset == 'finegym':
            frames = self._frame_reader.load_frames(
                video_name, start,
                start + self._clip_len * self._stride, pad=True,
                stride=self._stride, source_info = source_info)
        else:
            frames = self._frame_reader.load_frames(
                video_name, start, start + self._clip_len * self._stride, pad=True,
                stride=self._stride)

        return {'video': video_name, 'start': start // self._stride,
                'frame': frames}

    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        if self._dataset == 'soccernetball':
            labels_file = load_json(os.path.join(LABELS_SNB_PATH, meta['video'] + '/Labels-ball.json'))['annotations']
        else:
            labels_file = meta['events']
        
        num_frames = meta['num_frames']
        num_labels = math.ceil(num_frames / self._stride) 

        labels = np.zeros(num_labels, np.int64)
        for event in labels_file:
            if (self._dataset == 'soccernetball'):
                frame = int(int(event['position']) / 1000 * FPS_SNB)
            else:
                frame = event['frame']

            if (frame < num_frames):
                if event['label'] in self._class_dict: #Some labels in SoccerNetBall are not considered
                    labels[frame // self._stride] = self._class_dict[event['label']]
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))
        return labels

    @property
    def videos(self):
        if (self._dataset == 'soccernetball'):
            return sorted([
                (v['video'], math.ceil(v['num_frames'] / self._stride),
                FPS_SNB / self._stride) for v in self._labels])
        return sorted([
            (v['video'], math.ceil(v['num_frames'] / self._stride),
            v['fps'] / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                
                if (self._dataset == 'soccernetball'):
                    x_copy['fps'] = FPS_SNB / self._stride
                else:
                    x_copy['fps'] /= self._stride
                x_copy['num_frames'] //= self._stride


                if self._dataset == 'soccernetball':
                    labels_file = load_json(os.path.join(LABELS_SNB_PATH, x_copy['video'] + '/Labels-ball.json'))['annotations']
                    i = 0
                    while i < len(labels_file):
                        e = labels_file[i]
                        e['frame'] = int(int(e['position']) / 1000 * FPS_SNB) // self._stride
                        if e['label'] not in self._class_dict: 
                            labels_file.pop(i)
                        else:
                            i += 1
                            
                    x_copy['events'] = labels_file

                else:
                    for e in x_copy['events']:
                        e['frame'] //= self._stride

                labels.append(x_copy)
            return labels

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)

def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        print('{} : {} videos, {} frames'.format(
            src_file, len(labels), num_frames))

class FrameReader:

    def __init__(self, frame_dir, dataset):
        self._frame_dir = frame_dir
        self.dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path)
        return img
    
    def load_paths(self, video_name, start, end, stride=1, source_info = None):

        if self.dataset == 'finediving':
            video_name = video_name.replace('__', '/')
        
        if self.dataset == 'finegym':
            frame0 = source_info['start_frame'] - source_info['pad'][0]
            video_name = video_name.split('_')[0]  
            path = os.path.join(self._frame_dir, video_name)
        else:            
            path = os.path.join(self._frame_dir, video_name)

        found_start = -1
        pad_start = 0
        pad_end = 0
        for frame_num in range(start, end, stride):

            if frame_num < 0:
                pad_start += 1
                continue

            if pad_end > 0:
                pad_end += 1
                continue
            
            if self.dataset == 'finediving':
                frame = frame_num
                frame_path = os.path.join(path, f'%06d' % frame + '.jpg')
                base_path = path
                ndigits = 6

            elif self.dataset == 'f3set':
                frame = frame_num
                frame_path = os.path.join(path, f'%06d' % frame + '.jpg')
                base_path = path
                ndigits = 6   

            elif self.dataset == 'tennis':
                frame = frame_num
                frame_path = os.path.join(path, f'%06d' % frame + '.jpg')
                base_path = path
                ndigits = 6

            elif self.dataset == 'finegym':
                frame = frame0 + frame_num
                frame_path = os.path.join(path, f'%06d' % frame + '.jpg')
                base_path = path
                ndigits = 6

            elif self.dataset == 'soccernetball':
                frame = frame_num
                frame_path = os.path.join(path, 'frame' + str(frame) + '.jpg')
                base_path = path
                ndigits = -1
            
            exist_frame = os.path.exists(frame_path)
            if exist_frame & (found_start == -1):
                found_start = frame

            if not exist_frame:
                pad_end += 1

        ret = [base_path, found_start, pad_start, pad_end, ndigits, (end-start) // stride]

        return ret
    
    def load_frames(self, paths, pad=False, stride=1):
        base_path = paths[0]
        start = paths[1]
        pad_start = paths[2]
        pad_end = paths[3]
        ndigits = paths[4]
        length = paths[5]

        ret = []
        if ndigits == -1:
            path = os.path.join(base_path, 'frame')
            _ = [ret.append(self.read_frame(path + str(start + j * stride) + '.jpg')) for j in range(length - pad_start - pad_end)]

        else:
            path = base_path + '/'
            _ = [ret.append(self.read_frame(path + str(start + j * stride).zfill(ndigits) + '.jpg')) for j in range(length - pad_start - pad_end)]

        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if pad_start > 0 or (pad and pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))            

        return ret
    
class FrameReaderVideo:

    def __init__(self, frame_dir, dataset):
        self._frame_dir = frame_dir
        self._dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path)
        return img

    def load_frames(self, video_name, start, end, pad=False, stride=1, source_info = None):
        ret = []
        n_pad_start = 0
        n_pad_end = 0

        if self._dataset == 'finediving':
            video_name = video_name.replace('__', '/')

        if self._dataset == 'finegym':
            frame0 = source_info['start_frame'] - source_info['pad'][0]
            video_name = video_name.split('_')[0]  
            path = os.path.join(self._frame_dir, video_name)
        else:
            path = os.path.join(self._frame_dir, video_name)       

        for frame_num in range(start, end, stride):

            if frame_num < 0:
                n_pad_start += 1
                continue
            
            if self._dataset == 'finediving':
                frame_path = os.path.join(path, f'%06d' % (frame_num) + '.jpg')
            
            elif self._dataset == 'f3set':
                frame_path = os.path.join(path, f'%06d' % (frame_num) + '.jpg')

            elif self._dataset == 'soccernetball':
                frame_path = os.path.join(self._frame_dir, video_name, 'frame' + str(frame_num) + '.jpg')
            
            elif self._dataset == 'tennis':
                frame_path = os.path.join(path, f'%06d' % (frame_num) + '.jpg')

            elif self._dataset == 'finegym':
                frame_path = os.path.join(path, f'%06d' % (frame0 + frame_num) + '.jpg')
                
            try:
                img = self.read_frame(frame_path)
                ret.append(img)
            except RuntimeError:
                n_pad_end += 1

        if len(ret) == 0:
            return -1 # Return -1 if no frames were loaded

        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
        return ret