from app.view import app
import logging


def config_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('news.log', mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # 日志
    logger.debug('this is a logger debug message')
    logger.info('this is a logger info message')
    logger.warning('this is a logger warning message')
    logger.error('this is a logger error message')
    logger.critical('this is a logger critical message')


if __name__ == "__main__":
    # 设置日志
    # config_logger()
    app.run(port=9999)