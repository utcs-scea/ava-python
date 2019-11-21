#!/home/hyu/venv3/bin/python3

from absl import app
import os, sys, atexit, signal, time
import threading, queue

from ctypes import *
from structs import *
sys.path.insert(1, os.path.join(sys.path[0], '../include/common'))
from devconf import *
from tensorflow_py import *
import apifwd_tf
import apifwd_tflite

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../common/python'))
from cmd_channel import *

# Global
init_global()
cmd_queue = queue.Queue()    # Tensorflow
tflite_queue = queue.Queue() # Tensorflow Lite

# Register cleanup function
@atexit.register
def worker_exit():
    print("halt worker")

# Setup SIGINT and SIGSEGV handler
def signal_handler(sig, frame):
    print("segfault!")
    sys.exit(0)

#signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGSEGV, signal_handler)

# Check permission
if os.geteuid() != 0:
    print("need sudo permission")
    exit(-1)

# Check arguments
print(sys.executable, __file__, sys.argv[1:])
if len(sys.argv) <= 1:
    print("Usage: ./worker.py <worker_port>")
    print("Current argc = " + str(len(sys.argv)))
    exit(0)
[worker_port] = map(int, sys.argv[1:2])

# Create channel
# TODO: create different types of channel based on environment variables
chan = Command_channel_c()
chan.create_worker_channel(worker_port)
set_global_chan(chan)

# Spawn task-poller
def poll_task():
    print("worker@%d starts polling tasks" % worker_port)
    while True:
        recv_cmd = chan.receive_command()
        if recv_cmd.__get_api_id() == TENSORFLOW_PY_API:
            cmd_queue.put(recv_cmd)
            print("[worker@%d] new tf_py task" % (worker_port))
        elif recv_cmd.__get_api_id() == TENSORFLOW_LITE_API:
            tflite_queue.put(recv_cmd)
            print("[worker@%d] new tf_lite task" % (worker_port))
        else:
            print("[worker@%d] wrong runtime type" % worker_port)
        # FIXME: workaround for Queue.get() blocking issue

t = threading.Thread(target=poll_task, args=())
t.daemon = True
t.start()

# Execute handler in the main thread
apifwd_tf.handler(cmd_queue, chan)
#apifwd_tflite.handler(tflite_queue, chan)
