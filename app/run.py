from view import app
import logging
import os


def config_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('/var/log/Project/nlp_project.log', mode='w')
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
    if os.path.exists('/root/.flag'):
        config_logger()
    app.run(port=9999)