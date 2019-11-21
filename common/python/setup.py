from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

setup(
    cmdclass = { 'build_ext': build_ext },
    ext_modules = [ Extension("cmd_channel",
                              include_dirs = [
                                  np.get_include(),
                                  "../../include",
                                  "../../worker/include",
                                  "../../guestlib/include",
                                  "/usr/include/glib-2.0",
                                  "/usr/lib/x86_64-linux-gnu/glib-2.0/include"],
                              sources = [
                                  "cmd_channel.pyx",
                                  "../cmd_channel.c",
                                  "../cmd_channel_socket.c",
                                  "../cmd_channel_shm_worker.c",
                                  "../cmd_channel_shm.c",
                                  "../cmd_channel_osv_usetl.c",
                                  "../cmd_channel_record.c",
                                  "../cmd_channel_hv.c",
                                  "../zcopy.c",
                                  "../../worker/worker.c",
                                  "../../worker/manager_wrapper.c",
                                  "../cmd_handler.c",
                                  "../endpoint_lib.c",
                                  "../shadow_thread_pool.c",
                                  "../murmur3.c",
                                  "../socket.c",
                                  "../../guestlib/src/init.c"],
                              library_dirs = [
                                  "../../guestlib/build"],
                              libraries = [
                                  "glib-2.0",
                                  "usetl_device"],
                              define_macros = [("_GNU_SOURCE", None)],
                             ) ]
)
