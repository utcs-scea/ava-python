import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

from hook import *

def axy_computation(a, x, y):
  return a * x + y

inputs = [
    3.0,
    tf.ones([3, 3], tf.float32),
    tf.ones([3, 3], tf.float32),
]

tpu_computation = tpu.rewrite(axy_computation, inputs)

print("start")

tpu_grpc_url = TPUClusterResolver(
    tpu=[os.environ['TPU_NAME']]).get_master()

print(tpu_grpc_url)

sess = tf.Session(target=tpu_grpc_url)
sess.run(tpu.initialize_system())
sess.run(tf.global_variables_initializer())
output = sess.run(tpu_computation)
print(output)
sess.run(tpu.shutdown_system())

print('Done!')

