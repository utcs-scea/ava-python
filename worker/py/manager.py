#!/usr/bin/env /home/hyu/venv3.6-tf-test/bin/python3

'''
#!/usr/bin/env python3
'''

# Need upgrade to py3 for vsock

import os, sys, signal, select
import socket
import atexit
from structs import *

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../include'))
from devconf import *

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../common/python'))
from cmd_channel import *

# Register cleanup function
@atexit.register
def dispatcher_exit():
    print("halt dispatcher")


# Setup SIGINT handler
def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# Check permission
if os.geteuid() != 0:
    print("need sudo permission")
    exit(-1)

# Check Python
os.system('which python')

# Forcefully attaching socket to port 3333
listen_fd = Vsock_c()

# Initialize variables
msg = COMMAND_BASE()
response = COMMAND_BASE()
worker_id = 1

# Polling new applications
print("manager starts polling applications")
while 1:
    # get guestlib info
    msg = listen_fd.poll_client()

    if int(msg.__get_cmd_type()) == MSG_NEW_APPLICATION:
        pass

    else:
        print("[manager] wrong message type %d" % msg.command_type)
        listen_fd.close_client()
        continue

    # return worker port to guestlib
    listen_fd.respond_client(worker_id)

    # spawn a worker
    child = os.fork()
    if child == 0:
        listen_fd.close()
        break

    worker_id = worker_id + 1

# spawn worker
argv = [sys.executable,
        'worker.py', str(worker_id + WORKER_PORT_BASE)] + sys.argv[1:]
print("[manager] spawn worker --args=", argv)
os.execv(sys.executable, argv)
