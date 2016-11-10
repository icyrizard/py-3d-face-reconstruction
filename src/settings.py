"""
.. module:: settings_module
   :platform: Unix
   :synopsis: This module contains global settings.

"""
import logging
import logging.config
import os

LANDMARK_DETECTOR_PATH = '/data/shape_predictor_68_face_landmarks.dat'
#logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,  # this fixes the problem

    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname)s %(module)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': logging.INFO,
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'debug': {
            'level': logging.DEBUG,
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'root': {
            'handlers': ['default'],
            'level': logging.INFO,
            'propagate': True
        },
        'debug': {
            'handlers': ['debug'],
            'level': logging.DEBUG,
            'propagate': False
        }
    }
})

logger = logging.getLogger('root')
#logger.setLevel(logging.DEBUG)

if os.environ.get('DEBUG', False):
    logger = logging.getLogger('debug')
