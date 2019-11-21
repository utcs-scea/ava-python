import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes

import os, sys, fcntl, atexit, types

from init import *
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../include/common'))
from tensorflow_py import *
from devconf import *

init_global()
chan = init_guestlib()
set_global_chan(chan)

# test

def NwTestCallback(func, x, y):
    param, cb_param = InitTFParamWithCallback(TF_PY_NW_CALLBACK_TEST)
    serialized_func = SerializeMethod(func)
    dump1, buf1 = SerializeParam(param, param.param1, serialized_func)
    dump2, buf2 = SerializeParam(param, param.param2, x)
    dump3, buf3 = SerializeParam(param, param.param3, y)

    buf = ReserveReturnValue(param, param.ret_val1)

    IoctlWrapper(param, cb_param)

    return pickle.loads(buf)


# tensorflow

def NwGlobalVariablesInitializer():
    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), 0)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_GLOBAL_VARIABLES_INITIALIZER)
    cmd.send()
    ret_cmd = chan.receive_command()
    return NwObject(ret_cmd.__get_object_id())

tf.global_variables_initializer = NwGlobalVariablesInitializer

def NwOnes(shape, dtype=dtypes.float32, name=None):
    dump0, len0 = pickle_arg(shape)
    dump1, len1 = pickle_arg(dtype) if dtype != dtypes.float32 else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_ONES)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1)])
    cmd.send()
    ret_cmd = chan.receive_command()

    print("ones object_id=%d" % ret_cmd.__get_object_id())
    return NwObject(ret_cmd.__get_object_id())

tf.ones = NwOnes

def NwRandomUniform(shape,
                    minval=0,
                    maxval=None,
                    dtype=dtypes.float32,
                    seed=None,
                    name=None):
    dump1, len1 = pickle_arg(shape)
    dump2, len2 = pickle_arg(minval) if minval != 0 else (None, 0)
    dump3, len3 = pickle_arg(maxval) if maxval != None else (None, 0)
    dump4, len4 = pickle_arg(dtype) if dtype != dtypes.float32 else (None, 0)
    dump5, len5 = pickle_arg(seed) if seed != None else (None, 0)
    dump6, len6 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len1) + chan.buffer_size(len2) \
                      + chan.buffer_size(len3) + chan.buffer_size(len4) \
                      + chan.buffer_size(len5) + chan.buffer_size(len6)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_RANDOM_UNIFORM)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    offset4 = cmd.attach_buffer(dump4, len4)
    offset5 = cmd.attach_buffer(dump5, len5)
    offset6 = cmd.attach_buffer(dump6, len6)
    cmd.__set_tf_args([(len1, offset1), (len2, offset2), (len3, offset3),
                       (len4, offset4), (len5, offset5), (len6, offset6)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.random_uniform = NwRandomUniform

def NwTranspose(a, perm=None, name="transpose", conjugate=False):
    dump0, len0 = pickle_arg(a)
    dump1, len1 = pickle_arg(perm) if perm != None else (None, 0)
    dump2, len2 = pickle_arg(name) if name != "transpose" else (None, 0)
    dump3, len3 = pickle_arg(conjugate) if conjugate != False else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_TRANSPOSE)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.transpose = NwTranspose

def NwCast(x, dtype, name=None):
    dump0, len0 = pickle_arg(x)
    dump1, len1 = pickle_arg(dtype)
    dump2, len2 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_CAST)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.cast = NwCast

def NwExpandDims(input, axis=None, name=None, dim=None):
    dump0, len0 = pickle_arg(input)
    dump1, len1 = pickle_arg(axis) if axis != None else (None, 0)
    dump2, len2 = pickle_arg(name) if name != None else (None, 0)
    dump3, len3 = pickle_arg(dim) if dim != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_EXPAND_DIMS)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.expand_dims = NwExpandDims

def NwConcat(values, axis, name="concat"):
    dump0, len0 = pickle_arg(values)
    dump1, len1 = pickle_arg(axis)
    dump2, len2 = pickle_arg(name) if name != "concat" else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_CONCAT)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.concat = NwConcat

def NwEqual(x, y, name=None):
    dump0, len0 = pickle_arg(x)
    dump1, len1 = pickle_arg(y)
    dump2, len2 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_EQUAL)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.equal = NwEqual

def NwSlice(input_, begin, size, name=None):
    dump0, len0 = pickle_arg(input_)
    dump1, len1 = pickle_arg(begin)
    dump2, len2 = pickle_arg(size)
    dump3, len3 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_SLICE)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.slice = NwSlice

def NwShape(input_, name=None, out_type=dtypes.int32):
    dump0, len0 = pickle_arg(input_)
    dump1, len1 = pickle_arg(name) if name != None else (None, 0)
    dump2, len2 = pickle_arg(out_type) if out_type != dtypes.int32 else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_SHAPE)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.shape = NwShape

def NwFixedLenFeature(shape, dtype, default_value=None):
    dump0, len0 = pickle_arg(shape)
    dump1, len1 = pickle_arg(dtype)
    dump2, len2 = pickle_arg(default_value) if default_value != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_FIXED_LEN_FEATURE)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.FixedLenFeature = NwFixedLenFeature

def NwVarLenFeature(dtype):
    dump0, len0 = pickle_arg(dtype)
    total_buffer_size = chan.buffer_size(len0)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_VAR_LEN_FEATURE)
    offset0 = cmd.attach_buffer(dump0, len0)
    cmd.__set_tf_args([(len0, offset0)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.VarLenFeature = NwVarLenFeature

def NwParseSingleExample(serialized, features, name=None,
                         example_names=None):
    dump0, len0 = pickle_arg(serialized)
    dump1, len1 = pickle_arg(features)
    dump2, len2 = pickle_arg(name) if name != None else (None, 0)
    dump3, len3 = pickle_arg(example_names) if example_names != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_PARSE_SINGLE_EXAMPLE)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3)])
    cmd.send()
    ret_cmd = chan.receive_command()

    ret_val = unpickle_arg(ret_cmd, 0)
    return ret_val

tf.parse_single_example = NwParseSingleExample


# tensorflow.contrib.tpu

def NwInitializeSystem(embedding_config=None, job=None):
    dump0, len0 = pickle_arg(embedding_config) if embedding_config != None else (None, 0)
    dump1, len1 = pickle_arg(job) if job != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_TPU_INITIALIZE_SYSTEM)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1)])
    cmd.send()
    ret_cmd = chan.receive_command()
    print("initialize system object id", ret_cmd.__get_object_id())

    return NwObject(ret_cmd.__get_object_id())

tpu.initialize_system = NwInitializeSystem

'''
For a normal function, param.base.object_id returns the object id for the
remote object.
For a class method, param.base.object_id represents the object id for the
class object. The updated value represents the object id for the newly
created remote object.
'''
def NwShutdownSystem(job=None):
    dump0, len0 = pickle_arg(job) if job != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_TPU_SHUTDOWN_SYSTEM)
    offset0 = cmd.attach_buffer(dump0, len0)
    cmd.__set_tf_args([(len0, offset0)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tpu.shutdown_system = NwShutdownSystem

def NwRewrite(computation,
              inputs=None,
              infeed_queue=None,
              device_assignment=None,
              name=None):
    dump0, len0 = pickle_arg(computation)
    dump1, len1 = pickle_arg(inputs) if inputs != None else (None, 0)
    dump2, len2 = pickle_arg(infeed_queue) if infeed_queue != None else (None, 0)
    dump3, len3 = pickle_arg(device_assignment) if device_assignment != None else (None, 0)
    dump4, len4 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3) \
                      + chan.buffer_size(len4)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_TPU_REWRITE)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    offset4 = cmd.attach_buffer(dump4, len4)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3), (len4, offset4)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tpu.rewrite = NwRewrite

def NwRunConfig(tpu_config=None,
                evaluation_master=None,
                master=None,
                cluster=None,
                **kwargs):
    dump0, len0 = pickle_arg(tpu_config) if tpu_config != None else (None, 0)
    dump1, len1 = pickle_arg(evaluation_master) if evaluation_master != None else (None, 0)
    dump2, len2 = pickle_arg(master) if master != None else (None, 0)
    dump3, len3 = pickle_arg(cluster) if cluster != None else (None, 0)
    dump4, len4 = pickle_arg(kwargs)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3) \
                      + chan.buffer_size(len4)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_TPU_RUN_CONFIG)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    offset4 = cmd.attach_buffer(dump4, len4)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3), (len4, offset4)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.contrib.tpu.RunConfig = NwRunConfig

def NwTPUEstimator(model_fn=None,
                   model_dir=None,
                   config=None,
                   params=None,
                   use_tpu=True,
                   train_batch_size=None,
                   eval_batch_size=None,
                   predict_batch_size=None,
                   batch_axis=None,
                   eval_on_tpu=True,
                   export_to_tpu=True,
                   warm_start_from=None):
    dump0, len0 = pickle_arg(model_fn) if model_fn != None else (None, 0)
    dump1, len1 = pickle_arg(model_dir) if model_dir != None else (None, 0)
    dump2, len2 = pickle_arg(config) if config != None else (None, 0)
    dump3, len3 = pickle_arg(params) if params != None else (None, 0)
    dump4, len4 = pickle_arg(use_tpu) if use_tpu != True else (None, 0)
    dump5, len5 = pickle_arg(train_batch_size) if train_batch_size != None else (None, 0)
    dump6, len6 = pickle_arg(eval_batch_size) if use_tpu != None else (None, 0)
    dump7, len7 = pickle_arg(predict_batch_size) if use_tpu != None else (None, 0)
    dump8, len8 = pickle_arg(batch_axis) if use_tpu != None else (None, 0)
    dump9, len9 = pickle_arg(eval_on_tpu) if use_tpu != True else (None, 0)
    dump10, len10 = pickle_arg(export_to_tpu) if use_tpu != True else (None, 0)
    dump11, len11 = pickle_arg(warm_start_from) if use_tpu != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3) \
                      + chan.buffer_size(len4) + chan.buffer_size(len5) \
                      + chan.buffer_size(len6) + chan.buffer_size(len7) \
                      + chan.buffer_size(len8) + chan.buffer_size(len9) \
                      + chan.buffer_size(len10) + chan.buffer_size(len11)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_TPU_TPU_ESTIMATOR)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    offset4 = cmd.attach_buffer(dump4, len4)
    offset5 = cmd.attach_buffer(dump5, len5)
    offset6 = cmd.attach_buffer(dump6, len6)
    offset7 = cmd.attach_buffer(dump7, len7)
    offset8 = cmd.attach_buffer(dump8, len8)
    offset9 = cmd.attach_buffer(dump9, len9)
    offset10 = cmd.attach_buffer(dump10, len10)
    offset11 = cmd.attach_buffer(dump11, len11)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3), (len4, offset4), (len5, offset5),
                       (len6, offset6), (len7, offset7), (len8, offset8),
                       (len9, offset9), (len10, offset10), (len11, offset11)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.contrib.tpu.TPUEstimator = NwTPUEstimator


# Session

TfSession = tf.Session

#class NwSession(TfSession):
#    def __init__(self, target='', graph=None, config=None):
#        print("[hook] inside Session Init")
#        param = InitTFParam(TF_PY_SESSION_INIT)
#
#        if target != '':
#            dump1, buf1 = SerializeParam(param, param.param1, target)
#
#        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
#        if ret < 0:
#            print("ioctl_tf_cmd Session() failed: %d\n" % ret);
#            exit(-1);
#
#        self._object_id = param.base.object_id
#        if (self._object_id < 0):
#            return None
#
#    def __enter__(self):
#        param = InitTFParam(TF_PY_SESSION_ENTER)
#        param.base.object_id = self._object_id
#        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
#        if ret < 0:
#            print("ioctl_tf_cmd Session.__enter__ failed: %d\n" % ret);
#            exit(-1);
#        if self._object_id != param.base.object_id:
#            print("object IDs mismatch")
#            self._object_id = param.base.object_id
#        return self
#
#    def __exit__(self, exec_type, exec_value, exec_tb):
#        print(exec_type, exec_value, exec_tb)
#        return
#        param = InitTFParam(TF_PY_SESSION_EXIT)
#        param.base.object_id = self._object_id
#
#        dump1, buf1 = SerializeParam(param, param.param1, exec_type)
#        dump2, buf2 = SerializeParam(param, param.param2, exec_value)
#        dump3, buf3 = SerializeParam(param, param.param3, exec_tb)
#
#        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
#        if ret < 0:
#            print("ioctl_tf_cmd Session.__exit__ failed: %d\n" % ret);
#            exit(-1);
#
#    def run(self, fetches, feed_dict=None, options=None,
#            run_metadata=None):
#        print("run!")
#        param = InitTFParam(TF_PY_SESSION_RUN)
#        param.base.object_id = self._object_id
#
#        dump1, buf1 = SerializeParam(param, param.param1, fetches)
#
#        buf = ReserveReturnValue(param, param.ret_val1)
#
#        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
#        if ret < 0:
#            print("ioctl_tf_cmd Session.run failed: %d\n" % ret);
#            exit(-1);
#
#        return pickle.loads(buf)
#
#    def __del__(self):
#        param = InitTFParam(TF_PY_SESSION_DEL)
#        param.base.object_id = self._object_id
#        ret = fcntl.ioctl(fd, IOCTL_TF_PY_CMD, param, 1)
#        if ret < 0:
#            print("ioctl_tf_cmd Session.__del__ failed: %d\n" % ret);
#            exit(-1);

def NwSession2(target='', graph=None, config=None):
    dump0, len0 = pickle_arg(target) if target != '' else (None, 0)
    dump1, len1 = pickle_arg(graph) if graph != None else (None, 0)
    dump2, len2 = pickle_arg(config) if config != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_SESSION_INIT)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.Session = NwSession2


# TPUClusterResolver

def NwTPUClusterResolver(tpu=None,
                         zone=None,
                         project=None,
                         job_name='worker',
                         coordinator_name=None,
                         coordinator_address=None,
                         credentials='default',
                         service=None,
                         discovery_url=None):
    dump0, len0 = pickle_arg(tpu) if tpu != None else (None, 0)
    dump1, len1 = pickle_arg(zone) if zone != None else (None, 0)
    dump2, len2 = pickle_arg(project) if project != None else (None, 0)
    dump3, len3 = pickle_arg(job_name) if job_name != 'worker' else (None, 0)
    dump4, len4 = pickle_arg(coordinator_name) if coordinator_name != None else (None, 0)
    dump5, len5 = pickle_arg(coordinator_address) if coordinator_address != None else (None, 0)
    dump6, len6 = pickle_arg(credentials) if credentials != 'default' else (None, 0)
    dump7, len7 = pickle_arg(service) if service != None else (None, 0)
    dump8, len8 = pickle_arg(discovery_url) if discovery_url != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3) \
                      + chan.buffer_size(len4) + chan.buffer_size(len5) \
                      + chan.buffer_size(len6) + chan.buffer_size(len7) \
                      + chan.buffer_size(len8)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_TPU_CLUSTER_RESOLVER_INIT)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    offset4 = cmd.attach_buffer(dump4, len4)
    offset5 = cmd.attach_buffer(dump5, len5)
    offset6 = cmd.attach_buffer(dump6, len6)
    offset7 = cmd.attach_buffer(dump7, len7)
    offset8 = cmd.attach_buffer(dump8, len8)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3), (len4, offset4), (len5, offset5),
                       (len6, offset6), (len7, offset7), (len8, offset8)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.contrib.cluster_resolver.TPUClusterResolver = NwTPUClusterResolver
TPUClusterResolver = NwTPUClusterResolver


# control_flow_ops

def NwControlFlowOpsSwitch(data, pred, dtype=None, name=None):
    dump0, len0 = pickle_arg(data)
    dump1, len1 = pickle_arg(pred)
    dump2, len2 = pickle_arg(dtype) if dtype != None else (None, 0)
    dump3, len3 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_CONTROL_FLOW_OPS_SWITCH)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3)])
    cmd.send()
    ret_cmd = chan.receive_command()

    # returns a tuple (NwObject, index)
    ret_val = unpickle_arg(ret_cmd, 0)
    return ret_val

control_flow_ops.switch = NwControlFlowOpsSwitch

def NwControlFlowOpsMerge(inputs, name=None):
    dump0, len0 = pickle_arg(inputs)
    dump1, len1 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_CONTROL_FLOW_OPS_MERGE)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1)])
    cmd.send()
    ret_cmd = chan.receive_command()

    # returns a tuple (NwObject, index)
    ret_val = unpickle_arg(ret_cmd, 0)
    return ret_val

control_flow_ops.merge = NwControlFlowOpsMerge


# tf.image

def NwImageResizeImages(images,
                        size,
                        method=tf.image.ResizeMethod.BILINEAR,
                        align_corners=False,
                        preserve_aspect_ratio=False):
    dump0, len0 = pickle_arg(images)
    dump1, len1 = pickle_arg(size)
    dump2, len2 = pickle_arg(method) if method != tf.image.ResizeMethod.BILINEAR else (None, 0)
    dump3, len3 = pickle_arg(align_corners) if align_corners != False else (None, 0)
    dump4, len4 = pickle_arg(preserve_aspect_ratio) if preserve_aspect_ratio != False else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3) \
                      + chan.buffer_size(len4)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_IMAGE_RESIZE_IMAGES)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    offset4 = cmd.attach_buffer(dump4, len4)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3), (len4, offset4)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.image.resize_images = NwImageResizeImages

def NwImageSampleDistortedBoundingBox(image_size,
                                      bounding_boxes,
                                      seed=None,
                                      seed2=None,
                                      min_object_covered=0.1,
                                      aspect_ratio_range=None,
                                      area_range=None,
                                      max_attempts=None,
                                      use_image_if_no_bounding_boxes=None,
                                      name=None):
    dump0, len0 = pickle_arg(image_size)
    dump1, len1 = pickle_arg(bounding_boxes)
    dump2, len2 = pickle_arg(seed) if seed != None else (None, 0)
    dump3, len3 = pickle_arg(seed2) if seed2 != None else (None, 0)
    dump4, len4 = pickle_arg(min_object_covered) if min_object_covered != 0.1 else (None, 0)
    dump5, len5 = pickle_arg(aspect_ratio_range) if aspect_ratio_range != None else (None, 0)
    dump6, len6 = pickle_arg(area_range) if area_range != None else (None, 0)
    dump7, len7 = pickle_arg(max_attempts) if max_attempts != None else (None, 0)
    dump8, len8 = pickle_arg(use_image_if_no_bounding_boxes) if use_image_if_no_bounding_boxes != None else (None, 0)
    dump9, len9 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3) \
                      + chan.buffer_size(len4) + chan.buffer_size(len5) \
                      + chan.buffer_size(len6) + chan.buffer_size(len7) \
                      + chan.buffer_size(len8) + chan.buffer_size(len9)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_IMAGE_SAMPLE_DISTORTED_BOUNDING_BOX)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    offset4 = cmd.attach_buffer(dump4, len4)
    offset5 = cmd.attach_buffer(dump5, len5)
    offset6 = cmd.attach_buffer(dump6, len6)
    offset7 = cmd.attach_buffer(dump7, len7)
    offset8 = cmd.attach_buffer(dump8, len8)
    offset9 = cmd.attach_buffer(dump9, len9)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3), (len4, offset4), (len5, offset5),
                       (len6, offset6), (len7, offset7), (len8, offset8),
                       (len9, offset9)])
    cmd.send()
    ret_cmd = chan.receive_command()

    # returns a tuple (NwObject, NwObject, NwObject)
    ret_val = unpickle_arg(ret_cmd, 0)
    return ret_val

tf.image.sample_distorted_bounding_box = NwImageSampleDistortedBoundingBox

def NwImageDrawBoundingBoxes(images, boxes, name=None):
    dump0, len0 = pickle_arg(images)
    dump1, len1 = pickle_arg(boxes)
    dump2, len2 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_IMAGE_DRAW_BOUNDING_BOXES)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.image.draw_bounding_boxes = NwImageDrawBoundingBoxes

def NwImageDecodeJpeg(contents,
                      channels=0,
                      ratio=1,
                      fancy_upscaling=True,
                      try_recover_truncated=False,
                      acceptable_fraction=1,
                      dct_method="",
                      name=None):
    dump0, len0 = pickle_arg(contents)
    dump1, len1 = pickle_arg(channels) if channels != 0 else (None, 0)
    dump2, len2 = pickle_arg(ratio) if ratio != 1 else (None, 0)
    dump3, len3 = pickle_arg(fancy_upscaling) if fancy_upscaling != True else (None, 0)
    dump4, len4 = pickle_arg(try_recover_truncated) if try_recover_truncated != False else (None, 0)
    dump5, len5 = pickle_arg(acceptable_fraction) if acceptable_fraction != 1 else (None, 0)
    dump6, len6 = pickle_arg(dct_method) if dct_method != "" else (None, 0)
    dump7, len7 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3) \
                      + chan.buffer_size(len4) + chan.buffer_size(len5) \
                      + chan.buffer_size(len6) + chan.buffer_size(len7)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_IMAGE_DECODE_JPEG)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    offset4 = cmd.attach_buffer(dump4, len4)
    offset5 = cmd.attach_buffer(dump5, len5)
    offset6 = cmd.attach_buffer(dump6, len6)
    offset7 = cmd.attach_buffer(dump7, len7)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3), (len4, offset4), (len5, offset5),
                       (len6, offset6), (len7, offset7)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.image.decode_jpeg = NwImageDecodeJpeg

def NwImageConvertImageDtype(image, dtype, saturate=False, name=None):
    dump0, len0 = pickle_arg(image)
    dump1, len1 = pickle_arg(dtype)
    dump2, len2 = pickle_arg(saturate) if saturate != False else (None, 0)
    dump3, len3 = pickle_arg(name) if name != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2) + chan.buffer_size(len3)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_IMAGE_CONVERT_IMAGE_DTYPE)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    offset3 = cmd.attach_buffer(dump3, len3)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2),
                       (len3, offset3)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.image.convert_image_dtype = NwImageConvertImageDtype


# tf.data.Dataset

@staticmethod
def NwDataDatasetListFiles(file_pattern, shuffle=None, seed=None):
    dump0, len0 = pickle_arg(file_pattern)
    dump1, len1 = pickle_arg(shuffle) if shuffle != None else (None, 0)
    dump2, len2 = pickle_arg(seed) if seed != None else (None, 0)
    total_buffer_size = chan.buffer_size(len0) + chan.buffer_size(len1) \
                      + chan.buffer_size(len2)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_DATA_DATASET_LIST_FILES)
    offset0 = cmd.attach_buffer(dump0, len0)
    offset1 = cmd.attach_buffer(dump1, len1)
    offset2 = cmd.attach_buffer(dump2, len2)
    cmd.__set_tf_args([(len0, offset0), (len1, offset1), (len2, offset2)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

tf.data.Dataset.list_files = NwDataDatasetListFiles


# tf.keras

def NwKerasSequential(model):
    dump0, len0 = pickle_arg(model)
    total_buffer_size = chan.buffer_size(len0)

    cmd = chan.new_command(chan.__get_sizeof_tf_cmd(), total_buffer_size)
    cmd.__set_cmd_base(TENSORFLOW_PY_API, TF_PY_KERAS_SEQUENTIAL)
    offset0 = cmd.attach_buffer(dump0, len0);
    cmd.__set_tf_args([(len0, offset0)])
    cmd.send()
    ret_cmd = chan.receive_command()

    return NwObject(ret_cmd.__get_object_id())

# Common

@atexit.register
def close_fd():
    print("[hook] guestlib channel is closed")
