from os import environ
from os.path import join


def test_const(test_environment_vars):
    from const import DIR_TEMP, DIR_MODEL, MEMCACHE_IP, MEMCACHE_DETECTION_KEY, MEMCACHE_AUGMENTATION_KEY, \
        MEMCACHE_KEY_TIMEOUT, MEMCACHE_CAPTURE_KEY, RTSP_CAM_PORT, RTSP_CAM_IP, RTSP_CAM_LOGIN, RTSP_CAM_PASSWORD, \
        CAPTURE_SAVE_RESULT, CAPTURE_SLEEP, NO_DATA_PAUSE, DETECTOR_SLEEP, DETECTOR_SAVE_RESULT, AUGMENT_BBOX_COLOR, \
        AUGMENT_BBOX_TEXT_COLOR, AUGMENT_SAVE_RESULT, AUGMENT_SLEEP, DIR_ROOT
    assert DIR_TEMP == join(DIR_ROOT, 'temp')
    assert DIR_MODEL == join(DIR_ROOT, 'model_data')
    assert MEMCACHE_IP == '127.0.0.1:11211'
    assert MEMCACHE_DETECTION_KEY == 'detection'
    assert MEMCACHE_AUGMENTATION_KEY == 'result'
    assert MEMCACHE_CAPTURE_KEY == 'capture'
    assert MEMCACHE_KEY_TIMEOUT == 1234
    assert RTSP_CAM_PORT == '554'
    assert RTSP_CAM_IP == '10.171.18.13'
    assert RTSP_CAM_LOGIN == 'admin'
    assert RTSP_CAM_PASSWORD == 'qwerty12345'
    assert CAPTURE_SAVE_RESULT is True
    assert CAPTURE_SLEEP == 1234
    assert NO_DATA_PAUSE == 1234
    assert DETECTOR_SLEEP == 1234
    assert DETECTOR_SAVE_RESULT is True
    assert AUGMENT_BBOX_COLOR == (0, 255, 0)
    assert AUGMENT_BBOX_TEXT_COLOR == (0, 255, 0)
    assert AUGMENT_SAVE_RESULT is True
    assert AUGMENT_SLEEP == 1234


def test_const_env(test_environment_vars):
    environ['CAPTURE_SLEEP'] = '999'
    environ['RTSP_CAM_IP'] = '10.10.10.10'
    environ['CAPTURE_SAVE_RESULT'] = 'False'
    environ['AUGMENT_BBOX_COLOR'] = '[255, 255, 255]'
    from const import CAPTURE_SLEEP, RTSP_CAM_IP, CAPTURE_SAVE_RESULT, AUGMENT_BBOX_COLOR
    assert CAPTURE_SLEEP == 999
    assert RTSP_CAM_IP == '10.10.10.10'
    assert CAPTURE_SAVE_RESULT is False
    assert AUGMENT_BBOX_COLOR == (255, 255, 255)
