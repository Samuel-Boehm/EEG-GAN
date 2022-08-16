# Author: Samuel Böhm <samuel-boehm@web.de>

# Provides a logger initialization for the EEG-GAN project
import logging


class Formatter(logging.Formatter):
    """Formatter for logger to control the output depending on the chosen level"""

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = f"\033[{(97)}m%(levelname)s\033[0m: %(message)s"
        elif record.levelno == logging.DEBUG:
            self._style._fmt = f"\033[{(92)}m%(levelname)s\033[0m: [%(filename)s line: %(lineno)d] %(message)s"
        elif record.levelno == logging.WARNING:
            self._style._fmt = f"\033[{(91)}m%(levelname)s\033[0m: [%(filename)s] %(message)s"

        return super().format(record)


def set_level(logger, level='WARNING'):

    if level == 'INFO':
        logger.setLevel(level=logging.INFO)
    elif level == 'DEBUG':
        logger.setLevel(level=logging.DEBUG)
    elif level == 'WARNING':
        logger.setLevel(level=logging.WARNING)
    else:
        print(f'unknown level: {level} setting to WARNING')
        logger.setLevel(logging.WARNING)


def init_logger(logger, level='WARNING'):
    """Sets the logging formatter and level for a given logger"""

    set_level(logger=logger, level=level)

    handler = logging.StreamHandler()
    handler.setFormatter(Formatter())

    logger.addHandler(handler)



