import os
import time

t = time.time()
localtime = time.localtime(t)

mk_name = time.strftime("%Y%m%d_%H%M%S", localtime)
mk_path = os.path.join('/opt/code/', str(mk_name))
os.mkdir(mk_path)


os.system('git clone https://github.com/bianjing2018/Project.git {}'.format(mk_path))
if os.path.exists('/root/Project'):
    os.system('unlink /root/Project')

if os.path.exists('/root/init.py'):
    os.system('unlink /root/init.py')

if os.path.exists('/root/manager.sh'):
    os.system('unlink /root/manager.sh')


ln_manager_path = '/root/manager.sh'
ln_init_path = '/root/init.py'

origin_init_path = mk_path + '/shell/init.py'
origin_manager_path = mk_path + '/shell/manager.sh'

os.system('ln -s {} /root/Project'.format(mk_path))
os.system('ln -s {} {}'.format(origin_init_path, ln_init_path))
os.system('ln -s {} {}'.format(origin_manager_path, ln_manager_path))