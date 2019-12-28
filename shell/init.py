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
if os.path.exists('/root/Project'):
    os.system('unlink /root/Project')

ln_init_path = '/root/init.py'
if os.path.exists(ln_init_path):
    os.system('unlink {}'.format(ln_init_path))

ln_manager_path = '/root/manager.sh'
if os.path.exists(ln_manager_path):
    os.system('unlink {}'.format(ln_manager_path))


origin_init_path = os.path.join(mk_path, '/shell/init.py')
origin_manager_path = os.path.join(mk_path, '/shell/manager.sh')
os.system('ln -s {} /root/Project'.format(mk_path))
os.system('ln -s {} {}'.format(origin_init_path, ln_init_path))
os.system('ln -s {} {}'.format(origin_manager_path, ln_init_path))