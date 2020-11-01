from os.path import join
import logging
import cv2 as cv
from time import time, sleep
import memcache

from const import MEMCACHE_IP, MEMCACHE_CAPTURE_KEY, MEMCACHE_KEY_TIMEOUT, RTSP_CAM_CONNECTION_STRING, \
    CAPTURE_SLEEP, CAPTURE_SAVE_RESULT, DIR_TEMP

logger = logging.getLogger("stream_processing")


def frames_capture():
    vcap = cv.VideoCapture(RTSP_CAM_CONNECTION_STRING)
    connect = memcache.Client([MEMCACHE_IP])

    oper_id = 0
    oper_id_key = MEMCACHE_CAPTURE_KEY + '_id'
    while True:
        t = time()
        ret, frame = vcap.read()
        if frame is not None:
            encoded_frame = cv.imencode('.jpg', frame)[1].tostring()
            r = connect.set(MEMCACHE_CAPTURE_KEY, encoded_frame, MEMCACHE_KEY_TIMEOUT)
            if r:
                logger.debug(f'frame captured and stored in \"{MEMCACHE_CAPTURE_KEY}\" in {time() - t} secs')
                connect.set(oper_id_key, oper_id, MEMCACHE_KEY_TIMEOUT)
                oper_id += 1

            if CAPTURE_SAVE_RESULT:
                cv.imwrite(join(DIR_TEMP, 'capture.jpg'), frame)
        sleep(CAPTURE_SLEEP)
        cv.waitKey()

