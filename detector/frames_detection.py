from const import MEMCACHE_IP, MEMCACHE_CAPTURE_KEY, NO_DATA_PAUSE, DETECTOR_SLEEP, \
    DETECTOR_SAVE_RESULT, DIR_TEMP, MEMCACHE_DETECTION_KEY, MEMCACHE_DETECTION_TIME_KEY, DIR_MODEL, MEMCACHE_KEY_TIMEOUT

from os.path import join
import memcache
import cv2 as cv
from time import time, sleep
import numpy as np
import msgpack
import json
import logging

from detector.detector import YOLOv3

MODEL_NAME = join(DIR_MODEL, 'yolo.h5')
logger = logging.getLogger("stream_processing")


def frames_detection():
    connect = memcache.Client([MEMCACHE_IP])
    with open(join(DIR_MODEL, 'classes.json'), 'r') as f:
        classes = dict(json.loads(f.read()))
    detector = YOLOv3(MODEL_NAME, classes, 1)
    input_image_dims = tuple(map(int, detector.input_image_dims))

    oper_id = 0
    oper_id_key = MEMCACHE_DETECTION_KEY + '_id'
    capture_id_key = MEMCACHE_CAPTURE_KEY + '_id'
    while True:
        t = time()
        capture_oper_id = connect.get(capture_id_key)
        if capture_oper_id is not None:
            if capture_oper_id > oper_id:
                oper_id = capture_oper_id
                frame_bytes = connect.get(MEMCACHE_CAPTURE_KEY)
                if frame_bytes is not None:

                    frame = np.fromstring(frame_bytes, dtype=np.uint8)
                    frame = cv.imdecode(frame, cv.IMREAD_COLOR)
                    frame_h, frame_w = frame.shape[:2]
                    frame_resized = cv.resize(frame, input_image_dims)

                    normalized_data = detector.preprocess_array(frame_resized)
                    result = detector.single_detection(normalized_data)

                    classes = (result.get(0, {}).get('classes'))
                    bboxes = (result.get(0, {}).get('boxes'))

                    bboxes = bboxes * np.array([frame_w, frame_h, frame_w, frame_h])

                    result = msgpack.packb(dict(frame=frame_bytes,
                                                classes=classes.astype(int).tolist(),
                                                boxes=bboxes.astype(int).tolist()))
                    r = connect.set(MEMCACHE_DETECTION_KEY, result, MEMCACHE_KEY_TIMEOUT)
                    if r:
                        logger.debug(f'frame detected and stored in \"{MEMCACHE_DETECTION_KEY}\" in {time() - t} secs')
                        connect.set(oper_id_key, oper_id, MEMCACHE_KEY_TIMEOUT)
                        connect.set(MEMCACHE_DETECTION_TIME_KEY, time() - t, MEMCACHE_KEY_TIMEOUT)

                    if DETECTOR_SAVE_RESULT:
                        np.save(join(DIR_TEMP, 'classes'), classes)
                        np.save(join(DIR_TEMP, 'boxes'), bboxes)
                        cv.imwrite(join(DIR_TEMP, 'screenshot.jpg'), frame)

                else:
                    logger.debug(f'screenshot not found in memcache using \"{MEMCACHE_CAPTURE_KEY}\" key, '
                                 f'sleep {NO_DATA_PAUSE} secs')
                    sleep(NO_DATA_PAUSE)
        sleep(DETECTOR_SLEEP)
