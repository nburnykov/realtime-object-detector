import json
from os.path import join
import logging
import memcache
from time import time, sleep

import msgpack
import numpy as np
import cv2 as cv

from const import MEMCACHE_IP, MEMCACHE_DETECTION_KEY, NO_DATA_PAUSE, AUGMENT_BBOX_COLOR, AUGMENT_BBOX_TEXT_COLOR, \
    AUGMENT_SLEEP, AUGMENT_SAVE_RESULT, DIR_TEMP, DIR_MODEL, MEMCACHE_KEY_TIMEOUT, MEMCACHE_AUGMENTATION_KEY

logger = logging.getLogger("stream_processing")


def bbox_cut_out():
    connect = memcache.Client([MEMCACHE_IP])
    with open(join(DIR_MODEL, 'classes.json'), 'r') as f:
        classes_names = dict(json.loads(f.read()))
    i = 0
    while True:
        data_bytes = connect.get(MEMCACHE_DETECTION_KEY)
        if data_bytes is not None:
            data = msgpack.unpackb(data_bytes)
            bboxes = data.get(b'boxes')
            classes = data.get(b'classes')
            image_bytes = data.get(b'frame')

            frame = np.fromstring(image_bytes, dtype=np.uint8)
            frame = cv.imdecode(frame, cv.IMREAD_COLOR)

            for cl, bbx in zip(classes, bboxes):
                cl_name = classes_names.get(cl)
                if cl_name == 'person':
                    x1, y1 = bbx[:2]
                    x2, y2 = bbx[2:]
                    ROI = frame[y1:y2, x1:x2]
                    cv.imwrite(join(DIR_TEMP, f'person{i}.jpg'), ROI)
                    i += 1
