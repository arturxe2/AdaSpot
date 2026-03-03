# Global imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
from collections import defaultdict
from tabulate import tabulate
from itertools import groupby
import os
import zipfile
from SoccerNet.Evaluation.utils import LoadJsonFromZip
from SoccerNet.Evaluation.ActionSpotting import average_mAP
import json

# Local imports
from util.constants import TOLERANCES, TOLERANCES_SNB, WINDOWS, WINDOWS_SNB, INFERENCE_BATCH_SIZE, GAMES_SNB
from util.score import compute_mAPs
from util.io import store_json, store_json_snb

class ErrorStat:

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get(self):
        return self._err / self._total

    def get_acc(self):
        return 1. - self._get()
    
class ForegroundF1:

    def __init__(self):
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)

    def update(self, true, pred):
        if pred != 0:
            if true != 0:
                self._tp[None] += 1
            else:
                self._fp[None] += 1

            if pred == true:
                self._tp[pred] += 1
            else:
                self._fp[pred] += 1
                if true != 0:
                    self._fn[true] += 1
        elif true != 0:
            self._fn[None] += 1
            self._fn[true] += 1

    def get(self, k):
        return self._f1(k)

    def tp_fp_fn(self, k):
        return self._tp[k], self._fp[k], self._fn[k]

    def _f1(self, k):
        denom = self._tp[k] + 0.5 * self._fp[k] + 0.5 * self._fn[k]
        if denom == 0:
            assert self._tp[k] == 0
            denom = 1
        return self._tp[k] / denom
        
def process_frame_predictions(dataset, classes, pred_dict, high_recall_score_threshold=0.01):
    
    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}
    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    err = ErrorStat()
    f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}
    for video, (scores, support) in (sorted(pred_dict.items())):
        label = dataset.get_labels(video)
        if np.min(support) == 0:
            support[support == 0] = 1
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        pred = np.argmax(scores, axis=1)
        err.update(label, pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        for i in range(pred.shape[0]):
            f1.update(label[i], pred[i])

            if pred[i] != 0:
                events.append({
                    'label': classes_inv[pred[i]],
                    'frame': i,
                    'score': scores[i, pred[i]].item()
                })

            for j in classes_inv:
                if scores[i, j] >= high_recall_score_threshold:
                    events_high_recall.append({
                        'label': classes_inv[j],
                        'frame': i,
                        'score': scores[i, j].item()
                    })

        pred_events.append({
            'video': video, 'events': events,
            'fps': fps_dict[video]})
        pred_events_high_recall.append({
            'video': video, 'events': events_high_recall,
            'fps': fps_dict[video]})
        
    return err, f1, pred_events, pred_events_high_recall, pred_scores

def non_maximum_supression(pred, window, threshold = 0.0):
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            if type(window) is not list:
                class_window = window
            else:
                class_window = window[i]
                i += 1
            while(len(v) > 0):
                e1 = max(v, key=lambda x:x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                v.pop(pos1)
                list_pos = [pos for pos, e in enumerate(v) if ((e['frame'] >= e1['frame']-class_window) & (e['frame'] <= e1['frame']+class_window))]
                for pos in list_pos[::-1]: #reverse order to avoid movement of positions in the list
                    v.pop(pos)

        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred

def soft_non_maximum_supression(pred, window, threshold = 0.01):
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            if type(window) is not list:
                class_window = window
            else:
                class_window = window[i]
                i += 1
            while(len(v) > 0):
                e1 = max(v, key=lambda x:x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                list_pos = [pos for pos, e in enumerate(v) if ((e['frame'] >= e1['frame']-class_window) & (e['frame'] <= e1['frame']+class_window))]
                for pos in list_pos:
                    v[pos]['score'] = v[pos]['score'] * (np.abs(e1['frame'] - v[pos]['frame'])) ** 2 / ((class_window+0) ** 2)
                v.pop(pos1)

        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred

def downsample(array, stride = 2):
    if len(array) % stride != 0:
        array = array[:-(len(array) % stride)]
    
    return array.reshape(-1, stride).max(axis = 1)

# For F3Set to use their same Edit Score function, get_labels_start_end_time and levenstein functions from their codebase.
def edit_score(recognized, ground_truth, sets=[], norm=True, bg_class=[0]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm, sets)

def get_labels_start_end_time(frame_wise_labels, bg_class=[0]):
    labels = []
    starts = []
    ends = []
    if len(frame_wise_labels) <= 0:
        return labels, starts, ends
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends

def levenstein(p, y, norm=False, sets=[]):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            elif {y[j - 1], p[i - 1]} in sets:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                            D[i, j - 1] + 1,
                            D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def evaluate(model, dataset, split, classes, save_pred=None, printed = True, 
            test = False):
    
    tolerances = TOLERANCES
    windows = WINDOWS

    if dataset._dataset == 'soccernetball':
        tolerances = TOLERANCES_SNB
        windows = WINDOWS_SNB

    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))

    batch_size = INFERENCE_BATCH_SIZE
    num_workers = 8

    for clip in tqdm(DataLoader(
            dataset, num_workers=num_workers, pin_memory=True,
            batch_size=batch_size, prefetch_factor=1,
    )):

        # Batched by dataloader
        batch_pred_cls, batch_pred_scores = model.predict(clip['frame'])

        for i in range(clip['frame'].shape[0]):
            video = clip['video'][i]
            scores, support = pred_dict[video]
            pred_scores = batch_pred_scores[i]
            start = clip['start'][i].item()
            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0
            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores
            support[start:end] += (pred_scores.sum(axis=1) != 0) * 1

    err, f1, pred_events, pred_events_high_recall, pred_scores = \
        process_frame_predictions(dataset, classes, pred_dict, high_recall_score_threshold=0.01)

    # If not test, just compute mAP with NMS (0.10) and return.
    if not test:
        pred_events_high_recall = non_maximum_supression(pred_events_high_recall, window = windows[0], threshold = 0.10)
        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall, tolerances=tolerances, printed = True)
        avg_mAP = np.mean(mAPs)
        return avg_mAP
    
    # If test, compute mAP with NMS and SNMS, print results, and store predictions.
    else:
        
        # Results without postprocessing
        print('=== Results on {} (w/o NMS) ==='.format(split))
        print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

        def get_f1_tab_row(str_k):
            k = classes[str_k] if str_k != 'any' else None
            return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
        rows = [get_f1_tab_row('any')]
        for c in sorted(classes):
            rows.append(get_f1_tab_row(c))

        print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                        floatfmt='0.2f'))
        print()

        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall, tolerances=tolerances, printed = printed)
        avg_mAP = np.mean(mAPs)

        # Results with NMS
        print('=== Results on {} (w/ NMS{}) ==='.format(split, str(windows[0])))
        pred_events_high_recall_nms = non_maximum_supression(pred_events_high_recall, window = windows[0], threshold=0.01)
        mAPs, tolerances = compute_mAPs(dataset.labels, pred_events_high_recall_nms, tolerances=tolerances, printed = printed)
        avg_mAP_nms = np.mean(mAPs)

        # Results with SNMS
        print('=== Results on {} (w/ SNMS{}) ==='.format(split, str(windows[1])))
        pred_events_high_recall_snms = soft_non_maximum_supression(pred_events_high_recall, window = windows[1], threshold=0.01)
        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall_snms, tolerances=tolerances, printed = printed)
        avg_mAP_snms = np.mean(mAPs)

        # For F3Set, also compute event F1 and edit score
        if dataset._dataset == 'f3set':
            delta = 0
            f1_event = {}
            edit_scores = []
            pred_events = non_maximum_supression(pred_events, window = 10, threshold=0.01)
            for video_pred in pred_events:
                video_name = video_pred['video']
                labels = dataset.get_labels(video_name)
                preds_sequence = np.zeros_like(labels)
                preds = video_pred['events']
                for e in preds:
                    preds_sequence[e['frame']] = classes[e['label']]

                labels = downsample(labels, stride = 2) # To match downsampled predictions in F3Set / F3ED
                preds_sequence = downsample(preds_sequence, stride = 2)
                    
                # event F1 scores
                for i in range(len(preds_sequence)):
                    if preds_sequence[i] > 0 and preds_sequence[i] in labels[max(0, i - delta):min(len(preds_sequence), i + delta + 1)]:
                        if preds_sequence[i] not in f1_event:
                            f1_event[preds_sequence[i]] = [1, 0, 0]
                        else:
                            f1_event[preds_sequence[i]][0] += 1
                    if preds_sequence[i] > 0 and sum(labels[max(0, i - delta):min(len(preds_sequence), i + delta + 1)]) == 0:
                        if preds_sequence[i] not in f1_event:
                            f1_event[preds_sequence[i]] = [0, 1, 0]
                        else:
                            f1_event[preds_sequence[i]][1] += 1
                    if labels[i] > 0 and labels[i] not in preds_sequence[max(0, i - delta):min(len(preds_sequence), i + delta + 1)]:
                        if labels[i] not in f1_event:
                            f1_event[labels[i]] = [0, 0, 1]
                        else:
                            f1_event[labels[i]][2] += 1


                    gt = [k for k, g in groupby(labels) if k != 0]
                    pred = [k for k, g in groupby(preds_sequence) if k != 0]

                    edit_s = edit_score(pred, gt)
                    edit_scores.append(edit_s)

            f1, count = 0, 0
            for value in f1_event.values():
                if sum(value) == 0:
                    continue
                precision = value[0] / (value[0] + value[1] + 1e-10)
                recall = value[0] / (value[0] + value[2] + 1e-10)
                f1 += 2 * precision * recall / (precision + recall + 1e-10)
                count += 1
            f1 /= count

            print('Event F1 score: {:0.2f}'.format(f1 * 100))
            print('Edit score: {:0.2f}'.format(np.mean(edit_scores)))


        print('Storing predictions with SNMS')
        pred_events_high_recall_store = pred_events_high_recall_snms
            
        if save_pred is not None:
            if not os.path.exists('/'.join(save_pred.split('/')[:-1])):
                os.makedirs('/'.join(save_pred.split('/')[:-1]))
            store_json(save_pred + '.json', pred_events_high_recall_store)
                
            if dataset._dataset == 'soccernetball':
                store_json_snb(save_pred, pred_events_high_recall_store, stride = dataset._stride)

        return mAPs, tolerances
        
def process_frame_predictions(dataset, classes, pred_dict, high_recall_score_threshold=0.01):
    
    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}
    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    err = ErrorStat()
    f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}
    h = 0
    for video, (scores, support) in (sorted(pred_dict.items())):
        label = dataset.get_labels(video)
        if np.min(support) == 0:
            support[support == 0] = 1
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        pred = np.argmax(scores, axis=1)
        err.update(label, pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        for i in range(pred.shape[0]):
            f1.update(label[i], pred[i])

            if pred[i] != 0:
                events.append({
                    'label': classes_inv[pred[i]],
                    'frame': i,
                    'score': scores[i, pred[i]].item()
                })

            for j in classes_inv:
                if scores[i, j] >= high_recall_score_threshold:
                    events_high_recall.append({
                        'label': classes_inv[j],
                        'frame': i,
                        'score': scores[i, j].item()
                    })

        pred_events.append({
            'video': video, 'events': events,
            'fps': fps_dict[video]})
        pred_events_high_recall.append({
            'video': video, 'events': events_high_recall,
            'fps': fps_dict[video]})
        
    return err, f1, pred_events, pred_events_high_recall, pred_scores

def evaluate_SNB(label_path, pred_path, split = 'test', metric = 'at1', classes = None):

    return aux_evaluate(label_path, pred_path, list_games = GAMES_SNB[split], prediction_file = 'results_spotting.json',
            metric = metric, label_files = 'Labels-ball.json', framerate=25, classes = classes)

def aux_evaluate(SoccerNet_path, Predictions_path, list_games, prediction_file="results_spotting.json",
            framerate=2, metric="loose", label_files="Labels-v2.json", classes = None):

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    num_classes = len(classes)

    EVENT_DICTIONARY = {k:v-1 for k,v in classes.items()}        

    for game in tqdm(list_games):

        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, label_files))
        else:
            labels = json.load(open(os.path.join(SoccerNet_path, game, label_files)))
        
        # convert labels to vector
        label = label2vector(
            labels, num_classes=num_classes, EVENT_DICTIONARY=EVENT_DICTIONARY, framerate=framerate)

        # Load predictions
        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
        else:
            predictions = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
        # convert predictions to vector
        prediction = predictions2vector(
            predictions, num_classes=num_classes, EVENT_DICTIONARY=EVENT_DICTIONARY,
            framerate=framerate)

        targets_numpy.append(label)
        detections_numpy.append(prediction)

        closest_numpy = np.zeros(label.shape) - 1
        # Get the closest action index
        for c in np.arange(label.shape[-1]):
            indexes = np.where(label[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = label[indexes[i], c]
        closests_numpy.append(closest_numpy)

    # Define tolerances for mAP computation
    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])  # np.arange(1)*1 + 1
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])
    
    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible,
        "a_mAP_per_class_visible": a_mAP_per_class_visible,
        "a_mAP_unshown": a_mAP_unshown,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown,
    }
    return results

def label2vector(labels, num_classes=17, framerate=2, EVENT_DICTIONARY={}):

    vector_size = 120*60*framerate

    label_vec = np.zeros((vector_size, num_classes))

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        # annotation at millisecond precision
        if "position" in annotation:
            frame = int(framerate * ( int(annotation["position"])/1000 ))
        # annotation at second precision
        else:
            frame = framerate * ( seconds + 60 * minutes )

        if event not in EVENT_DICTIONARY:
            continue
        label = EVENT_DICTIONARY[event]

        # Label to 1 for visible actions, -1 for not visible ones --> in SoccerNet evaluation joint and visible/non-visible metrics are reported
        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        frame = min(frame, vector_size-1)
        label_vec[frame][label] = value

    return label_vec

def predictions2vector(predictions, num_classes=17, framerate=2, EVENT_DICTIONARY={}):

    vector_size = 120*60*framerate

    prediction = np.zeros((vector_size, num_classes))-1

    for annotation in predictions["predictions"]:

        time = int(annotation["position"])
        event = annotation["label"]

        frame = int(framerate * ( time/1000 ))

        if event not in EVENT_DICTIONARY:
            continue
        label = EVENT_DICTIONARY[event]

        value = annotation["confidence"]

        frame = min(frame, vector_size-1)
        prediction[frame][label] = value

    return prediction