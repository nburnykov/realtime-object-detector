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


def frames_augmentation():
    connect = memcache.Client([MEMCACHE_IP])
    with open(join(DIR_MODEL, 'classes.json'), 'r') as f:
        classes_names = dict(json.loads(f.read()))

    oper_id = 0
    oper_id_key = MEMCACHE_DETECTION_KEY + '_id'
    detect_id_key = MEMCACHE_DETECTION_KEY + '_id'
    while True:
        t = time()
        detect_oper_id = connect.get(detect_id_key)
        if detect_oper_id is not None:
            if detect_oper_id > oper_id:
                oper_id = detect_oper_id
                data_bytes = connect.get(MEMCACHE_DETECTION_KEY)
                if data_bytes is not None:
                    data = msgpack.unpackb(data_bytes)
                    bboxes = data.get(b'boxes')
                    classes = data.get(b'classes')
                    image_bytes = data.get(b'frame')

                    frame = np.fromstring(image_bytes, dtype=np.uint8)
                    frame = cv.imdecode(frame, cv.IMREAD_COLOR)

                    for cl, bbx in zip(classes, bboxes):
                        first = tuple(bbx[:2])
                        second = tuple(bbx[2:])
                        frame = cv.rectangle(frame, first, second, AUGMENT_BBOX_COLOR, 1)
                        frame = cv.putText(frame, classes_names.get(cl), first, cv.FONT_HERSHEY_PLAIN, 1,
                                           AUGMENT_BBOX_TEXT_COLOR)

                    frame_encoded = cv.imencode('.jpg', frame)[1].tostring()
                    r = connect.set(MEMCACHE_AUGMENTATION_KEY, frame_encoded, MEMCACHE_KEY_TIMEOUT)
                    if r:
                        logger.debug(f'frame augmented and stored in \"{MEMCACHE_AUGMENTATION_KEY}\" in {time() - t} secs')
                        connect.set(oper_id_key, oper_id, MEMCACHE_KEY_TIMEOUT)

                    if AUGMENT_SAVE_RESULT:
                        cv.imwrite(join(DIR_TEMP, 'capture_bboxes.jpg'), frame)

                else:
                    logger.debug(f'data not found in memcache using \"{MEMCACHE_DETECTION_KEY}\" key, '
                                 f'sleep {NO_DATA_PAUSE} secs')
                    sleep(NO_DATA_PAUSE)
        sleep(AUGMENT_SLEEP)


