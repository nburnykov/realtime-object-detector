from os import getenv
from os.path import dirname, abspath, join

from load_var import C

DIR_ROOT = dirname(abspath(__file__))
DIR_CONF = join(DIR_ROOT, 'config')

ENV_VAR_NAME = 'APP_CONFIG'
app_config = getenv(ENV_VAR_NAME)
config = C(join(DIR_CONF, app_config))

DIR_TEMP = join(DIR_ROOT, config.load_variable('DIR_TEMP', ''))
DIR_MODEL = join(DIR_ROOT, config.load_variable('DIR_MODEL', ''))

MEMCACHE_IP = config.load_variable('MEMCACHE_IP', '')

MEMCACHE_CAPTURE_KEY = config.load_variable('MEMCACHE_CAPTURE_KEY', '')
MEMCACHE_DETECTION_KEY = config.load_variable('MEMCACHE_DETECTION_KEY', '')
MEMCACHE_DETECTION_TIME_KEY = config.load_variable('MEMCACHE_DETECTION_TIME_KEY', '')
MEMCACHE_AUGMENTATION_KEY = config.load_variable('MEMCACHE_AUGMENTATION_KEY', '')
MEMCACHE_KEY_TIMEOUT = config.load_variable('MEMCACHE_KEY_TIMEOUT', var_type=float, default_value=0)  # seconds

RTSP_CAM_LOGIN = config.load_variable('RTSP_CAM_LOGIN', '')
RTSP_CAM_PASSWORD = config.load_variable('RTSP_CAM_PASSWORD', '')
RTSP_CAM_IP = config.load_variable('RTSP_CAM_IP', '')
RTSP_CAM_PORT = config.load_variable('RTSP_CAM_PORT', '')
connection = dict(RTSP_CAM_LOGIN=RTSP_CAM_LOGIN, RTSP_CAM_PASSWORD=RTSP_CAM_PASSWORD, RTSP_CAM_IP=RTSP_CAM_IP,
                  RTSP_CAM_PORT=RTSP_CAM_PORT)
RTSP_CAM_CONNECTION_STRING = config.load_variable('RTSP_CAM_CONNECTION_STRING', '').format(**connection)

CAPTURE_SLEEP = config.load_variable('CAPTURE_SLEEP', var_type=float, default_value=0)
CAPTURE_SAVE_RESULT = config.load_variable('CAPTURE_SAVE_RESULT', var_type=bool,
                                           default_value=False)  # for the purposes of debugging, saves result on HDD
NO_DATA_PAUSE = config.load_variable('NO_DATA_PAUSE', var_type=float, default_value=2)  # seconds

DETECTOR_SLEEP = config.load_variable('DETECTOR_SLEEP', var_type=float, default_value=0)  # seconds
DETECTOR_SAVE_RESULT = config.load_variable('DETECTOR_SAVE_RESULT', var_type=bool,
                                            default_value=False)  # for the purposes of debugging, saves result on HDD

AUGMENT_BBOX_COLOR = tuple(config.load_variable('AUGMENT_BBOX_COLOR', var_type='json', default_value=[0, 0, 0]))
AUGMENT_BBOX_TEXT_COLOR = tuple(
    config.load_variable('AUGMENT_BBOX_TEXT_COLOR', var_type='json', default_value=[0, 0, 0]))
AUGMENT_SAVE_RESULT = config.load_variable('AUGMENT_SAVE_RESULT', var_type=bool,
                                           default_value=False)  # for the purposes of debugging, saves result on HDD
AUGMENT_SLEEP = config.load_variable('AUGMENT_SLEEP', var_type=float, default_value=1)  # seconds

LOGGING_CONF = config.load_variable('LOGGING_CONF', var_type='json', default_value={})

try:
    from local_config import *
except ImportError:
    pass
