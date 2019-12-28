import os
import time

t = time.time()
localtime = time.localtime(t)

mk_name = time.strftime("%Y%m%d_%H%M%S", localtime)
mk_path = os.path.join('/opt/code/', str(mk_name))
if os.path.exists(mk_path):
    os.remove(mk_path)
else:
    os.mkdir(mk_path)

os.system('git clone https://github.com/bianjing2018/Project.git {}'.format(mk_path))
if os.path.exists('/opt/Project'):
    os.system('unlink /opt/Project')

if os.path.exists('/root/init.py'):
    os.system('unlink /root/init.py')

if os.path.exists('/root/manager.sh'):
    os.system('/root/manager.sh')

os.system('ln -s {} /root/Project'.format(mk_path))
os.system('ln-s {}/shell/init.py /root/init.py')
os.system('ln-s {}/shell/manager.sh /root/manager.sh')