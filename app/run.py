from view import app
import logging
import os
from tools.LoggingConfig import config_logger


# from app.models import create_app, db
# from app.models import User, Role
# from flask.ext.script import Manager, Shell
#
#
# app = create_app()
# manager = Manager(app)
#
#
# def make_shell_context():
#     return dict(app=app, db=db, User=User, Role=Role)
#
#
# manager.add_command("shell", Shell(make_context=make_shell_context))


if __name__ == "__main__":
    # 设置日志
    if os.path.exists('/root/.flag'):
        config_logger()
    app.run(port=9999)