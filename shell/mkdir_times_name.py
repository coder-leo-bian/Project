import os
import time

t = time.time()
localtime = time.localtime(t)
mk_name = time.strftime("%Y%m%d_%H%M%S", localtime)
if os.path.exists('/opt/code/' + mk_name):
    os.remove('/opt/code/' + mk_name)
else:
    os.mkdir('/opt/code', mk_name)
