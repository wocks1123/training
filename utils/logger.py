import logging

from logging import handlers

from datetime import timezone, timedelta, datetime

kst = timezone(timedelta(hours=9))
start_time = str(datetime.now(tz=kst).strftime("%Y-%m-%d_%H_%M_%S"))
log_filename = './data/log/{}_{}.log'.format(start_time, "LOG_TITLE")

class Logger(object):
    def __init__(self, fmt='[%(asctime)s][%(levelname)s]: %(message)s'):
        self.logger = logging.getLogger()
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        self.logger.addHandler(sh)

        th = handlers.TimedRotatingFileHandler(
            filename=log_filename,
            when='D',
            backupCount=3
        )
        th.setFormatter(format_str)
        self.logger.addHandler(th)

    def info(self, info):
        self.logger.info(info)

    def debug(self, info):
        self.logger.debug(info)

    def warning(self, info):
        self.logger.warning(info)

    def exception(self, info):
        self.logger.exception(info)


logger = Logger()
def log_config(config):
    for key in config:
        if isinstance(config[key], dict):
            for k in config[key]:
                logger.info('{}: {}'.format(k, config[key][k]))
        else:
            logger.info('{}: {}'.format(key, config[key]))
    logger.info('---------------------------------load config done---------------------------------')
