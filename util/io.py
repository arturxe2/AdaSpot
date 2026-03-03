# Global imports
import os
import json

# Local imports
from util.constants import FPS_SNB

def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)
    
def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines

def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)

def store_json_inference(out_path, pred, stride = 1):
    predDict = dict()
    predDict['predictions'] = []
    for event in pred['events']:
        eventDict = dict()
        frame = int(event['frame']) * stride
        eventDict['frame'] = frame
        eventDict['label'] = event['label']
        eventDict['confidence'] = event['score']
        predDict['predictions'].append(eventDict)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(out_path + '/results_inference.json', 'w') as fp:
        json.dump(predDict, fp, indent=4)

def store_json_snb(pred_path, pred, stride = 1):
    for game in pred:
        gameDict = dict()
        gameDict['UrlLocal'] = game['video']
        gameDict['predictions'] = []
        for event in game['events']:
            eventDict = dict()
            position = int(event['frame'] / FPS_SNB * 1000 * stride)
            eventDict['gameTime'] = '1 - {}:{}'.format(position // 60000, int((position % 60000) // 1000))
            eventDict['label'] = event['label']
            eventDict['position'] = position
            eventDict['confidence'] = event['score']
            eventDict['half'] = 1
            gameDict['predictions'].append(eventDict)

        path = os.path.join('/'.join(pred_path.split('/')[:-1]) + '/preds', game['video'])
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/results_spotting.json', 'w') as fp:
            json.dump(gameDict, fp, indent=4)