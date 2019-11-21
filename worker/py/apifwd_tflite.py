import sys, os
import traceback
import threading, queue
from functools import reduce
import operator

from edgetpu.classification.engine import ClassificationEngine
from edgetpu.detection.engine import DetectionEngine

from structs import *
sys.path.insert(1, os.path.join(sys.path[0], '../include/common'))
from devconf import *
from tensorflow_py import *

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../common/python'))
from cmd_channel import *

initialized = False

def is_unpickleable_type(obj):
    #if isinstance(obj,
    #        (tf.Session, tf.Tensor, tf.Operation, tf.data.Dataset,
    #         tf.SparseTensor, tf.data.Iterator, tf.contrib.tpu.TPUEstimator)):
    #    return True
    return False


def tuple_mapper(t, index_list):
    global object_dict
    global object_id

    if (t is None) or (not isinstance(t, tuple)):
        return t

    l = list(t)
    for index in index_list:
        obj = l[index]
        if is_unpickleable_type(obj):
            object_dict[object_id] = obj
            l[index] = NwObject(object_id)
            object_id += 1

    return tuple(l)


def dict_mapper(d):
    global object_dict
    global object_id

    if (d is None) or (type(d) is not dict):
        return
    for k, v in d.items():
        if isinstance(v, dict):
            dict_mapper(d[k])
        else:
            if is_unpickleable_type(v):
                object_dict[object_id] = v
                d[k] = NwObject(object_id)
                object_id += 1


def list_mapper(l):
    global object_dict
    global object_id

    if (l is None) or (not isinstance(l, list)):
        return
    for i in range(len(l)):
        if isinstance(l[i], list):
            list_mapper(l[i])
        else:
            if is_unpickleable_type(l[i]):
                object_dict[object_id] = l[i]
                l[i] = NwObject(object_id)
                object_id += 1


def tuple_walker(t):
    if (t is None) or (not isinstance(t, tuple)):
        return

    l = list(t)
    for i in range(len(l)):
        if isinstance(l[i], tuple):
            l[i] = tuple_walker(l[i])
        else:
            if type(l[i]) is NwObject:
                l[i] = object_dict[l[i].object_id()]
                print("tanslate in tuple:", l[i])

    return tuple(l)


def list_walker(l):
    if (l is None) or (not isinstance(l, list)):
        return
    for i in range(len(l)):
        if isinstance(l[i], list):
            list_walker(l[i])
        else:
            if type(l[i]) is NwObject:
                l[i] = object_dict[l[i].object_id()]
                print("tanslate in list:", l[i])


def dict_walker(d):
    if (d is None) or (type(d) is not dict):
        return
    for k, v in d.items():
        if isinstance(v, dict):
            dict_walker(d[k])
        elif isinstance(v, list):
            list_walker(d[k])
        else:
            if isinstance(v, NwObject):
                d[k] = object_dict[v.object_id()]
                print("tanslate in dict:", d[k])


def handler(cmd_queue, chan):
    global object_dict
    global object_id
    global callback_stack

    global initialized
    if not initialized:
        callback_stack = []
        object_dict = dict()
        object_id = 1
        initialized = True
        print("handler is initialized")

    while True:
        cmd = cmd_queue.get(block=True)
        cmd_id = cmd.__get_cmd_id()
        print("new command id %d" % cmd_id)

        try:
            if cmd_id == TF_LITE_CLASSIFICATION_ENGINE:
                param0 = unpickle_arg(cmd, 0)
                print(param0)
                engine = ClassificationEngine(param0)

                # assign object_id
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = engine
                ret_cmd.__set_object_id(object_id)
                ret_cmd.__set_cmd_base(TENSORFLOW_LITE_API, TF_LITE_CLASSIFICATION_ENGINE_RET)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_LITE_DETECTION_ENGINE:
                param0 = unpickle_arg(cmd, 0)
                print(param0)
                engine = DetectionEngine(param0)

                # assign object_id
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = engine
                ret_cmd.__set_object_id(object_id)
                ret_cmd.__set_cmd_base(TENSORFLOW_LITE_API, TF_LITE_DETECTION_ENGINE_RET)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_NW_OBJECT:
                obj = object_dict[cmd.__get_object_id()]
                name = unpickle_arg(cmd, 0)
                args = unpickle_arg(cmd, 1)
                kwargs = unpickle_arg(cmd, 2)
                print("NwObject", obj, name, args, kwargs)

                # expand embedded NwObject
                args = list(args)
                list_walker(args)
                args = tuple(args)
                dict_walker(kwargs)
                print("after translation", obj, name, args, kwargs)

                # run
                result = getattr(obj, name)(*(args or []), **(kwargs or {}))
                print("analyze type", type(result), result)

                # TODO: go through tuple, dict or list
                if isinstance(result, tuple):
                    result = tuple_mapper(result, range(len(result)))
                if isinstance(result, dict):
                    dict_mapper(result)
                if isinstance(result, list):
                    list_mapper(result)

                ret_cmd = None

                # serialize return value
#                if isinstance(result, list):
                    # Check whether a nested list pickles
                    # https://github.com/uqfoundation/dill/issues/307
#                    pickleable = pickle.pickles(reduce(operator.add, result))
#                else:
#                    pickleable = pickle.pickles(result)
                pickleable = True
                if is_unpickleable_type(result) or not pickleable:
                    ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                    object_dict[object_id] = result
                    ret_cmd.__set_object_id(object_id)
                    object_id += 1

                else:
                    dump_ret, len_ret = pickle_arg(result)
                    total_buffer_size = chan.buffer_size(len_ret)
                    ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
                    ret_cmd.__set_cmd_base(TENSORFLOW_LITE_API, TF_PY_NW_OBJECT_RET)
                    offset_ret = ret_cmd.attach_buffer(dump_ret, len_ret)
                    ret_cmd.__set_object_id(0)
                    ret_cmd.__set_tf_args([(len_ret, offset_ret)])

                ret_cmd.send()

            else:
                print("unsupported Tensorflow API %d" % cmd_id)

        except Exception as error:
            print("fault: ", str(error))
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            traceback.print_stack()

        cmd.free_command()
        print("finished cmd %d" % cmd_id)
