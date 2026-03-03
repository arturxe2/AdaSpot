'''
We define constants here that are used across the codebase.
'''

LABELS_SNB_PATH = '/snb/labels/path'
STRIDE = 1
STRIDE_SNB = 2
EVAL_SPLITS = ['test']
OVERLAP = 0.9
F3SET_ELEMENTS = [2, 3, 3, 3, 7, 8, 2, 4]
DEFAULT_PAD_LEN = 5
FPS_SNB = 25

GAMES_SNB = {
        'train': ["england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
            "england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday",
            "england_efl/2019-2020/2019-10-01 - Brentford - Bristol City",
            "england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest"],
        'val' : ["england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End"],
        'test': ["england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town",
            "england_efl/2019-2020/2019-10-01 - Reading - Fulham"],
        'challenge': ["england_efl/2019-2020/2019-10-02 - Cardiff City - Queens Park Rangers",
            "england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City"]
        }

# Evaluation
TOLERANCES = [0, 1, 2, 4]
TOLERANCES_SNB = [6, 12] # To match 0.5 seconds and 1 second at 25 fps (with stride of 2) in SoccerNet
WINDOWS = [1, 2] # window for nms and snms (2 for soft-nms as detailed in the paper)
WINDOWS_SNB = [6, 12]
INFERENCE_BATCH_SIZE = 4

