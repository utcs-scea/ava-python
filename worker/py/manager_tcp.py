#!/usr/bin/env python3

# Need upgrade to py3 for vsock

import os, sys, signal, select
import socket
import atexit
from structs import *

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../include'))
from devconf import *

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

# Forcefully attaching socket to port 4000
listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR | socket.SO_REUSEPORT, 1)
listen_sock.bind(('localhost', 4000))
listen_sock.listen(10)

# Initialize variables
msg = COMMAND_BASE()
response = COMMAND_BASE()
worker_id = 1
guest_cid = 0

# Polling new applications
print("manager starts polling applications")
while 1:
    (client_sock, addr) = listen_sock.accept()

    # get guestlib info
    recv_buf = client_sock.recv(sizeof(COMMAND_BASE))
    msg = COMMAND_BASE.from_buffer_copy(recv_buf)
    if int(msg.command_type) == MSG_NEW_APPLICATION:
        pass

    else:
        print("[manager] wrong message type %d" % msg.command_type)
        client_sock.close()
        continue

    # return worker port to guestlib
    response.api_id = INTERNAL_API
    response.command_size = sizeof(COMMAND_BASE)
    response.data_size = 0
    response.reserved_area.uint64[0] = worker_id + WORKER_PORT_BASE
    client_sock.send(response)
    client_sock.close()

    # spawn a worker
    child = os.fork()
    if child == 0:
        listen_sock.close()
        break

    worker_id = worker_id + 1

# spawn worker
argv = [sys.executable,
        'worker.py',
        str(worker_id + WORKER_PORT_BASE)] + sys.argv[1:]
print("[manager] spawn worker --args=", argv)
os.execv(sys.executable, argv)
