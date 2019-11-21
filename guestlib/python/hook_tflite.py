import edgetpu
from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine

import os, sys, fcntl, atexit, types

from init import *
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../include/common'))
from tensorflow_py import *
from devconf import *

init_global()
chan = init_guestlib()
set_global_chan(chan)

# classification

def AvaClassificationEngine(model_path):
    dump0, len0 = pickle_arg(model_path)
    total_buffer_size = chan.buffer_size(len0)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_LITE_API, TF_LITE_CLASSIFICATION_ENGINE)
    offset0 = cmd.attach_buffer(dump0, len0)
    cmd.__set_tf_args([(len0, offset0)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id(), api_id = TENSORFLOW_LITE_API)

edgetpu.classification.engine.ClassificationEngine = AvaClassificationEngine
ClassificationEngine = AvaClassificationEngine

def AvaDetectionEngine(model_path):
    dump0, len0 = pickle_arg(model_path)
    total_buffer_size = chan.buffer_size(len0)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_LITE_API, TF_LITE_DETECTION_ENGINE)
    offset0 = cmd.attach_buffer(dump0, len0)
    cmd.__set_tf_args([(len0, offset0)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id(), api_id = TENSORFLOW_LITE_API)

DetectionEngine = AvaDetectionEngine
edgetpu.detection.engine.DetectionEngine = AvaDetectionEngine

# Common

@atexit.register
def close_fd():
    print("[hook] guestlib channel is closed")
