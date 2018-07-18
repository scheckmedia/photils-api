import logging
import os
from logging.handlers import RotatingFileHandler
from config import Config


# https://stackoverflow.com/questions/7621897/python-logging-module-globally
def setup_custom_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel.get(Config.get('log').get('level', 'INFO')))

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if Config.get('log').get('file', None):
        path = Config.get('log').get('file')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        max_bytes = Config.get('log').get('max_size', 5) * 1024 * 1024
        backup_count = Config.get('log').get('backup_count', 5)
        rotating_handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        rotating_handler.setFormatter(formatter)
        logger.addHandler(rotating_handler)

    return logger