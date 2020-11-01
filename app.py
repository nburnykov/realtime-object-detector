from flask import Response
from flask import Flask
from flask import render_template

import memcache
from time import sleep
import cv2 as cv
import numpy as np

from multiprocessing import Process
from detector.frames_capture import frames_capture
from detector.frames_detection import frames_detection
from detector.frames_augmentation import frames_augmentation

from const import MEMCACHE_IP, MEMCACHE_AUGMENTATION_KEY, LOGGING_CONF, MEMCACHE_DETECTION_TIME_KEY

import logging
from logging.config import dictConfig

logger = logging.getLogger("stream_processing")
dictConfig(LOGGING_CONF)

logger.info('Service loading')

app = Flask(__name__)
connect = memcache.Client([MEMCACHE_IP])

capture = Process(target=frames_capture)
detection = Process(target=frames_detection)
augmentation = Process(target=frames_augmentation)

capture.start()
detection.start()
augmentation.start()

logger.info('Service loaded')


@app.route("/")
def index():
    return render_template("index.html")


def generate():
    while True:
        image_bytes = connect.get(MEMCACHE_AUGMENTATION_KEY)
        detection_time = connect.get(MEMCACHE_DETECTION_TIME_KEY)

        if image_bytes is None:
            continue

        frame = np.fromstring(image_bytes, dtype=np.uint8)
        frame = cv.imdecode(frame, cv.IMREAD_COLOR)

        (flag, encoded_image) = cv.imencode(".jpg", frame)

        if not flag:
            continue
        print(1 / detection_time)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# @app.route("/detection_fps")
# def detection_fps():
#     return Response(tuple(generate())[1])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True,
            threaded=True, use_reloader=False)
