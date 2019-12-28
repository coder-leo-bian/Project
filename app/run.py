from view import app
import logging
import os
from tools.LoggingConfig import config_logger


if __name__ == "__main__":
    # 设置日志
    if os.path.exists('/root/.flag'):
        config_logger()
    app.run(port=9999)