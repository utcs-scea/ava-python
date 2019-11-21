import sys, os, fcntl, time
import traceback
import threading, queue
from functools import reduce
import operator

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import tpu
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

# FIXME: a hack for callback
from stub import *

from structs import *
sys.path.insert(1, os.path.join(sys.path[0], '../include/common'))
from devconf import *
from tensorflow_py import *

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../common/python'))
from cmd_channel import *

initialized = False


def parse_param(vm_id, mm, param, payload):
    if payload.size == 0:
        return None

    print("parse_param: vm_id=%d, cmd=%d, obj=%d, dstore=%lx,offset=%x, payload_size=%d, payload_offset=%d" %
          (vm_id,
           param.base.cmd_id,
           param.base.object_id,
           param.base.dstore_size, param.base.dstore_offset,
           payload.size, payload.offset))

    mm.seek(INVOKER_FIFO_SIZE + VGPU_DSTORE_SIZE * (vm_id - 1) +
            param.base.dstore_offset + payload.offset)
    param_dump = mm.read(payload.size)
    print("load:", pickle.loads(param_dump))
    return pickle.loads(param_dump)


def writeback_result(vm_id, mm, param, payload, arg):
    print("writeback", arg, "at dstore_offset =", param.base.dstore_offset, \
          "offset =", payload.offset, "size =", payload.size)
    dump = pickle.dumps(arg)
    payload.size = len(dump)
    print("writeback dump size =", payload.size)
    mm.seek(INVOKER_FIFO_SIZE +
            VGPU_DSTORE_SIZE * (vm_id - 1) +
            param.base.dstore_offset +
            payload.offset)
    mm.write(dump)


def is_unpickleable_type(obj):
    if isinstance(obj,
            (tf.Session, tf.Tensor, tf.Operation, tf.data.Dataset,
             tf.SparseTensor, tf.data.Iterator, tf.contrib.tpu.TPUEstimator)):
        return True
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


def callback_constructor(callback_id, callback_param, param, mm, vm_id,
                         cmd_queue, kvm_fd):
    class_id = param.base.class_id

    def request_callback(*args, **kwargs):
        global callback_stack

        writeback_result(vm_id, mm,
                         callback_param,
                         callback_param.param1,
                         args);
        writeback_result(vm_id, mm,
                         callback_param,
                         callback_param.param2,
                         kwargs);
        print("request callback id=%d" % callback_id)
        callback_param.base.object_id = callback_id
        param.base.done = STATUS_TASK_CALLBACK

        callback_stack.append({"callback_id":callback_id,
                               "param": param,
                               "callback_param": callback_param,
                               "deadline": time.time() + 20})

#        # spawn a new handler for TF APIs in the callback
#        t_cb_tf = threading.Thread(target = handler,
#                                   args = (cmd_queue, kvm_fd, mm))
#        t_cb_tf.start()
#
#        # spin until done becomes RUNNING
#        spin_deadline = time.time() + 20
#        while time.time() < spin_deadline:
#            if param.base.done == STATUS_TASK_RUNNING:
#                print("callback finishes normally")
#                break
#            time.sleep(0.2)
#        print("callback loop exits")
#
#        # stop the callback thread
#        node = MINI_TASK_NODE()
#        node.vm_id = STOP_HANDLER
#        cmd_queue.put(node)
#        t_cb_tf.join()
#        print("callback thread is shutdown")

        # handle in-callback APIs
        status = handler(cmd_queue, kvm_fd, mm)

        callback_stack.pop()

        # validate and copy callback result
        if param.base.done != STATUS_TASK_RUNNING or \
           status != STATUS_CALLBACK_DONE:
            print("callback state error, state=%d, status=%d" %
                  param.base.done, status)
            return -1
        else:
            # TODO: update args and kwargs
            cb_ret = parse_param(vm_id, mm,
                                 callback_param,
                                 callback_param.ret_val1)
            print("receive callback result", cb_ret)
            if isinstance(cb_ret, list):
                list_walker(cb_ret)
            elif isinstance(cb_ret, dict):
                dict_walker(cb_ret)
            elif isinstance(cb_ret, tuple):
                cb_ret = tuple_walker(cb_ret)
            print("translated callback result", cb_ret)

            return cb_ret

    def tpu_estimator_callback(params):
        return request_callback(params)

    print("class_id=%d" % class_id)
    if class_id == TF_PY_TPU_TPU_ESTIMATOR:
        return tpu_estimator_callback
    else:
        return request_callback

def handler(cmd_queue, chan):
    global object_dict
    global object_id
    global callback_stack

    global initialized
    if not initialized:
        callback_stack = []
        object_dict = dict()
        object_id = 1
        # TODO: forward logging or disable it in test
        tf.logging.set_verbosity(tf.logging.INFO)
        initialized = True
        print("handler is initialized")

    while True:
        cmd = cmd_queue.get(block=True)
        cmd_id = cmd.__get_cmd_id()
        print("new command id %d" % cmd_id)

        try:
            if cmd_id == TF_PY_NW_CALLBACK_DONE:
                param.base.done = STATUS_TASK_DONE
                ret = fcntl.ioctl(kvm_fd, IOCTL_KVM_NOTIFY_TASK_FINISHED, task.node_id)
                if ret < 0:
                    print("notify task completion failed: %d\n" % ret);
                if callback_stack and \
                   callback_stack[-1]["callback_id"] == param.base.object_id:
                    print("callback is finished")
                    return STATUS_CALLBACK_DONE
                else:
                    print("callback is error")
                    return STATUS_CALLBACK_ERROR

            if cmd_id == TF_PY_SESSION_INIT:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                sess = tf.Session(param0, param1, param2)

                # assign object_id
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = sess
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            #elif cmd_id == TF_PY_SESSION_ENTER:
            #    sess = object_dict[param.base.object_id]
            #    ctx_sess = sess.__enter__()
            #    if sess is ctx_sess:
            #        pass
            #    else: # unlikely
            #        print("unlikely to search for sess")
            #        param.base.object_id = next(obj_id for obj_id, obj in
            #                object_dict.items() if obj is ctx_sess)

            #elif cmd_id == TF_PY_SESSION_EXIT:
            #    param1 = parse_param(vm_id, mm, param, param.param1)
            #    param2 = parse_param(vm_id, mm, param, param.param2)
            #    param3 = parse_param(vm_id, mm, param, param.param3)

            #    sess = object_dict[param.base.object_id]
            #    sess.__exit__(param1, param2, param3)

            #elif cmd_id == TF_PY_SESSION_DEL:
            #    sess = object_dict[param.base.object_id]
            #    sess.__del__()

            # deprecated
            #elif cmd_id == TF_PY_SESSION_RUN:
            #    sess = object_dict[param.base.object_id]
            #    param1 = parse_param(vm_id, mm, param, param.param1)

            #    if type(param1) == NwObject:
            #        print("get NwObject=%d" % param1.object_id())
            #        param1 = object_dict[param1.object_id()]
            #        print(param1)

            #    ret_val = sess.run(param1)
            #    print(ret_val)

            #    writeback_result(vm_id, mm, param, param.ret_val1, ret_val);

            elif cmd_id == TF_PY_TPU_CLUSTER_RESOLVER_INIT:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                print("TPUClusterResolver", param0, param1, param2)
                tpu_grpc = tf.contrib.cluster_resolver.TPUClusterResolver(
                        tpu=param0, zone=param1, project=param2)

                # assign object_id
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = tpu_grpc
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            # deprecated
            elif cmd_id == TF_PY_TPU_CLUSTER_RESOLVER_MASTER:
                tpu_grpc = object_dict[cmd.__get_object_id()]
                # FIXED: may have parameters
                tpu_grpc_url = tpu_grpc.master()

                # serialize return value
                dump_ret, len_ret = pickle_arg(tpu_grpc_url)
                total_buffer_size = chan.buffer_size(len_ret)
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
                ret_cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_TPU_CLUSTER_RESOLVER_MASTER_RET)
                offset_ret = ret_cmd.attach_buffer(dump_ret, len_ret)
                ret_cmd.__set_tf_args([(len_ret, offset_ret)])
                ret_cmd.send()

            elif cmd_id == TF_PY_TPU_INITIALIZE_SYSTEM:
                # TODO: may have parameters
                ts = tpu.initialize_system()

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = ts
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_TPU_SHUTDOWN_SYSTEM:
                # TODO: may have parameters
                ts = tpu.shutdown_system()

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = ts
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_GLOBAL_VARIABLES_INITIALIZER:
                # TODO: may have parameters
                ts = tf.global_variables_initializer()

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = ts
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_ONES:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                print(param0)
                if param1 is None:
                    param1 = dtypes.float32
                print(param1)
                var = tf.ones(param0, param1)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = var
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_RANDOM_UNIFORM:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                param4 = unpickle_arg(cmd, 4)
                param5 = unpickle_arg(cmd, 5)
                if param1 is None:
                    param1 = 0
                if param3 is None:
                    param3 = dtypes.float32
                print(param0, param1, param2, param3)
                var = tf.random_uniform(param0, param1, param2, param3, param4, param5)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = var
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_TRANSPOSE:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                param0 = object_dict[param0.object_id()]
                if param2 is None:
                    param2 = "transpose"
                if param3 is None:
                    param3 = False
                print("transpose", param0, param1, param2, param3)
                var = tf.transpose(param0, param1, param2, param3)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = var
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_CAST:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param0 = object_dict[param0.object_id()]
                print("cast", param0, param1, param2)
                var = tf.cast(param0, param1, param2)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = var
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_EXPAND_DIMS:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                param0 = object_dict[param0.object_id()]
                print("expand_dims", param0, param1, param2, param3)
                var = tf.expand_dims(param0, param1, param2, param3)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = var
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_CONCAT:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param0 = object_dict[param0.object_id()]
                if param2 is None:
                    param2 = "concat"
                print("concat", param0, param1, param2)
                var = tf.concat(param0, param1, param2)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = var
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_EQUAL:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param0 = object_dict[param0.object_id()]
                print("equal", param0, param1, param2)
                if isinstance(param1, NwObject):
                    param1 = object_dict[param1.object_id()]
                result = tf.equal(param0, param1, param2)
                print(result)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = result
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_FIXED_LEN_FEATURE:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)

                feature = tf.FixedLenFeature(param0, param1, param2)
                print(feature)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = feature
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_VAR_LEN_FEATURE:
                param0 = unpickle_arg(cmd, 0)

                feature = tf.VarLenFeature(param0)
                print(feature)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = feature
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_PARSE_SINGLE_EXAMPLE:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                print(param1, param2)

                # expand embedded NwObject
                if isinstance(param0, NwObject):
                    param0 = object_dict[param0.object_id()]
                dict_walker(param1)
                print("after translation", param0, param1)

                result = tf.parse_single_example(param0, param1, param2, param3)
                print(result)
                dict_mapper(result)
                print(result)

                dump_ret, len_ret = pickle_arg(result)
                total_buffer_size = chan.buffer_size(len_ret)
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
                ret_cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_PARSE_SINGLE_EXAMPLE_RET)
                offset_ret = ret_cmd.attach_buffer(dump_ret, len_ret)
                ret_cmd.__set_tf_args([(len_ret, offset_ret)])
                ret_cmd.send()

            elif cmd_id == TF_PY_CONTROL_FLOW_OPS_SWITCH:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                param0 = object_dict[param0.object_id()]
                param1 = object_dict[param1.object_id()]
                print("switch", param0, param1, param2, param3)
                result = control_flow_ops.switch(param0, param1, param2, param3)
                print(result)

                mapped_tuple = tuple_mapper(result, [0, 1])
                print(mapped_tuple)

                dump_ret, len_ret = pickle_arg(mapped_tuple)
                total_buffer_size = chan.buffer_size(len_ret)
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
                ret_cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_CONTROL_FLOW_OPS_SWITCH_RET)
                offset_ret = ret_cmd.attach_buffer(dump_ret, len_ret)
                ret_cmd.__set_tf_args([(len_ret, offset_ret)])
                ret_cmd.send()

            elif cmd_id == TF_PY_CONTROL_FLOW_OPS_MERGE:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param0 = object_dict[param0.object_id()]
                print("merge", param0, param1)
                list_walker(param0)
                print("merge-new", param0, param1)
                result = control_flow_ops.merge(param0, param1)
                print(result)

                mapped_tuple = tuple_mapper(result, [0])
                print(mapped_tuple)
                dump_ret, len_ret = pickle_arg(mapped_tuple)
                total_buffer_size = chan.buffer_size(len_ret)
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
                ret_cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_CONTROL_FLOW_OPS_MERGE_RET)
                offset_ret = ret_cmd.attach_buffer(dump_ret, len_ret)
                ret_cmd.__set_tf_args([(len_ret, offset_ret)])
                ret_cmd.send()

            elif cmd_id == TF_PY_TPU_REWRITE:
                # TODO: may have more parameters
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                # default parameter
                if param1 is None:
                    param1 = None
                # expand embedded NwObject
                list_walker(param1)
                func = tpu.rewrite(param0, param1)
                print("Rewrite result:", func, " object id =", object_id)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = func
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_TPU_RUN_CONFIG:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                param4 = unpickle_arg(cmd, 4)
                # default parameter
                if param0 is None:
                    param0 = None
                if param1 is None:
                    param1 = None
                if param2 is None:
                    param2 = None
                if param3 is None:
                    param3 = None

                # expand embedded NwObject
                param3 = object_dict[param3.object_id()]
                print(param3, param4)
                func = tpu.RunConfig(param0, param1, param2, param3, **param4)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = func
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_TPU_TPU_ESTIMATOR:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                param4 = unpickle_arg(cmd, 4)
                param5 = unpickle_arg(cmd, 5)
                param6 = unpickle_arg(cmd, 6)
                param7 = unpickle_arg(cmd, 7)
                param8 = unpickle_arg(cmd, 8)
                param9 = unpickle_arg(cmd, 9)
                param10 = unpickle_arg(cmd, 10)
                param11 = unpickle_arg(cmd, 11)
                # default parameter
                if param0 is None:
                    param0 = None
                if param1 is None:
                    param1 = None
                if param2 is None:
                    param2 = None
                if param3 is None:
                    param3 = None
                if param4 is None:
                    param4 = True
                if param5 is None:
                    param5 = None
                if param6 is None:
                    param6 = None
                if param7 is None:
                    param7 = None
                if param8 is None:
                    param8 = None
                if param9 is None:
                    param9 = True
                if param10 is None:
                    param10 = True
                if param11 is None:
                    param11 = None

                # expand embedded NwObject
                param2 = object_dict[param2.object_id()]
                print(param2)
                func = tpu.TPUEstimator(param0, param1, param2, param3, param4,
                                        param5, param6, param7, param8, param9,
                                        param10, param11)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = func
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_IMAGE_RESIZE_IMAGES:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                param4 = unpickle_arg(cmd, 4)
                # default parameter
                if param2 is None:
                    param2 =ResizeMethod.BILINEAR
                if param3 is None:
                    param3 = False
                if param4 is None:
                    param4 = False

                # expand embedded NwObject
                param0 = object_dict[param0.object_id()]
                print(param0)
                img = tf.image.resize_images(param0, param1, param2, param3, param4)

                # TODO: it may return a float
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = img
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_SLICE:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)

                # expand embedded NwObject
                print(param0, param1, param2, param3)
                param0 = object_dict[param0.object_id()]
                ret = tf.slice(param0, param1, param2, param3)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = ret
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_SHAPE:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                if param2 is None:
                    param2 = dtypes.int32

                # expand embedded NwObject
                print(param0, param1, param2)
                param0 = object_dict[param0.object_id()]
                ret = tf.shape(param0, param1, param2)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = ret
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_IMAGE_SAMPLE_DISTORTED_BOUNDING_BOX:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                param4 = unpickle_arg(cmd, 4)
                param5 = unpickle_arg(cmd, 5)
                param6 = unpickle_arg(cmd, 6)
                param7 = unpickle_arg(cmd, 7)
                param8 = unpickle_arg(cmd, 8)
                param9 = unpickle_arg(cmd, 9)
                # default parameter
                if param4 is None:
                    param4 = 0.1

                print("sample_distorted_bounding_box", param0, param1)
                result = tf.image.sample_distorted_bounding_box(
                        param0, param1, param2, param3, param4, param5, param6,
                        param7, param8, param9)
                print(result)

                mapped_tuple = tuple_mapper(result, [0, 1, 2])
                print(mapped_tuple)
                dump_ret, len_ret = pickle_arg(mapped_tuple)
                total_buffer_size = chan.buffer_size(len_ret)
                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
                ret_cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_CONTROL_FLOW_OPS_MERGE_RET)
                offset_ret = ret_cmd.attach_buffer(dump_ret, len_ret)
                ret_cmd.__set_tf_args([(len_ret, offset_ret)])
                ret_cmd.send()

            elif cmd_id == TF_PY_IMAGE_DRAW_BOUNDING_BOXES:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)

                # expand embedded NwObject
                print(param0, param1, param2)
                param0 = object_dict[param0.object_id()]
                param1 = object_dict[param1.object_id()]
                ret = tf.image.draw_bounding_boxes(param0, param1, param2)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = ret
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_IMAGE_DECODE_JPEG:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)
                param4 = unpickle_arg(cmd, 4)
                param5 = unpickle_arg(cmd, 5)
                param6 = unpickle_arg(cmd, 6)
                param7 = unpickle_arg(cmd, 7)

                if param1 is None:
                    param1 = 0
                if param2 is None:
                    param2 = 1
                if param3 is None:
                    param3 = True
                if param4 is None:
                    param4 = False
                if param5 is None:
                    param5 = 1
                if param6 is None:
                    param6 = ""
                param0 = object_dict[param0.object_id()]
                ret = tf.image.decode_jpeg(param0, param1, param2, param3,
                        param4, param5, param6, param7)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = ret
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_IMAGE_CONVERT_IMAGE_DTYPE:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)
                param3 = unpickle_arg(cmd, 3)

                # expand embedded NwObject
                print(param0, param1, param2, param3)
                param0 = object_dict[param0.object_id()]
                if param2 is None:
                    param2 = False
                ret = tf.image.convert_image_dtype(param0, param1, param2, param3)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = ret
                ret_cmd.__set_object_id(object_id)
                object_id += 1
                ret_cmd.send()

            elif cmd_id == TF_PY_DATA_DATASET_LIST_FILES:
                param0 = unpickle_arg(cmd, 0)
                param1 = unpickle_arg(cmd, 1)
                param2 = unpickle_arg(cmd, 2)

                print(param0, param1, param2)
                if isinstance(param0, NwObject):
                    param0 = object_dict[param0.object_id()]
                ret = tf.data.Dataset.list_files(param0, param1, param2)

                ret_cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
                object_dict[object_id] = ret
                ret_cmd.__set_object_id(object_id)
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
                    ret_cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_NW_OBJECT_RET)
                    offset_ret = ret_cmd.attach_buffer(dump_ret, len_ret)
                    ret_cmd.__set_object_id(0)
                    ret_cmd.__set_tf_args([(len_ret, offset_ret)])

                ret_cmd.send()

            elif cmd_id == TF_PY_NW_METHOD:
                # Reuse as callback

                #ins = parse_param(vm_id, mm, param, param.param1)
                #name = parse_param(vm_id, mm, param, param.param2)
                #print(ins, name)

                #method = getattr(ins, name)
                #print(method)
                #object_dict[object_id] = method

                cw = callback_constructor(object_id, callback_param,
                        param, mm, vm_id, cmd_queue, kvm_fd)
                object_dict[object_id] = cw
                param.base.object_id = object_id
                object_id += 1

            elif cmd_id == TF_PY_NW_CALLBACK_TEST:
                nw_func = parse_param(vm_id, mm, param, param.param1)
                print(nw_func, nw_func.object_id())
                func = object_dict[nw_func.object_id()]
                print("callback func", func)
                x = parse_param(vm_id, mm, param, param.param2)
                y = parse_param(vm_id, mm, param, param.param3)
                result = func(x, y)
                print(result)
                writeback_result(vm_id, mm, param, param.ret_val1, result);

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
