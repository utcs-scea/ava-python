import os, sys
import ctypes
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.string cimport strcpy, strlen, memcpy
cimport cmd_channel

cdef int nw_worker_id

cdef class Command_base_c:
    cdef cmd_channel.command_channel *chan
    cdef cmd_channel.command_base *cmd

    def __init__(self):
        pass

    cdef __set_ptr(self, cmd_channel.command_channel *c, cmd_channel.command_base *ptr):
        self.chan = c
        self.cmd = ptr

    def attach_buffer(self, buf, size):
        p = (ctypes.c_byte * len(buf)).from_buffer_copy(buf) if not buf is None else None
#        p = ctypes.create_string_buffer(buf, len(buf)) if not buf is None else None
        addr = ctypes.addressof(p) if not p is None else 0
        real_addr = <void *>(<uint64_t>addr)
        real_size = <uint64_t>size
        with nogil:
            buf_id = command_channel_attach_buffer(self.chan, self.cmd,
                    real_addr, real_size)
        return <int>buf_id

    def get_buffer(self, buffer_id):
        return <char *>command_channel_get_buffer(self.chan, self.cmd, <void *>buffer_id)

    def send(self):
        with nogil:
            command_channel_send_command(self.chan, self.cmd)

    def print_command(self):
        command_channel_print_command(self.chan, self.cmd)

    def free_command(self):
        with nogil:
            command_channel_free_command(self.chan, self.cmd)

    def __get_vm_id(self):
        return self.cmd.vm_id

    def __get_api_id(self):
        return self.cmd.api_id

    def __get_cmd_id(self):
        return self.cmd.command_id

    def __get_cmd_type(self):
        return self.cmd.command_type

    def __get_object_id(self):
        py_cmd = <py_tf_cmd *>self.cmd
        return py_cmd.object_id

    def __get_tf_arg(self, idx):
        py_cmd = <py_tf_cmd *>self.cmd
        buf_id = py_cmd.buffer_id[idx]
        buf_len = py_cmd.buffer_len[idx]
        if buf_len is 0:
            return None
        real_buf_id = <void *>(<uint64_t>buf_id)
        buf = (ctypes.c_byte * buf_len)()
        real_buf = <void *>(<uint64_t>(ctypes.addressof(buf)))
        real_buf_len = <uint64_t>buf_len
        with nogil:
            buf_addr = command_channel_get_buffer(self.chan, self.cmd, real_buf_id)
            memcpy(real_buf, buf_addr, real_buf_len)
        return bytes(buf)

    def __get_pb_offset(self):
        pb_info = <param_block_info *>self.cmd.reserved_area
        return <uintptr_t>pb_info.param_local_offset

    def __get_pb_size(self):
        pb_info = <param_block_info *>self.cmd.reserved_area
        return <uintptr_t>pb_info.param_block_size

    def __set_cmd_base(self, api_id, command_id):
        self.cmd.api_id = api_id
        self.cmd.command_id = command_id

    def __set_object_id(self, object_id):
        py_cmd = <py_tf_cmd *>self.cmd
        py_cmd.object_id = object_id

    def __set_tf_args(self, arg_list):
        py_cmd = <py_tf_cmd *>self.cmd
        lens, offsets = zip(*arg_list)
        for i in range(len(lens)):
            py_cmd.buffer_id[i] = <uintptr_t>offsets[i]
            py_cmd.buffer_len[i] = <uintptr_t>lens[i]

cdef class Command_channel_c:
    cdef cmd_channel.command_channel *chan
    cdef str chan_type

    def __init__(self):
        if "AVA_CHANNEL" in os.environ:
            self.chan_type = os.environ["AVA_CHANNEL"]
        else:
            self.chan_type = "LOCAL"

    def create_guestlib_channel(self):
        if self.chan_type == "LOCAL":
            with nogil:
                self.chan = command_channel_min_new()
        elif self.chan_type == "SHM":
            with nogil:
                self.chan = command_channel_shm_new()
        elif self.chan_type == "VSOCK":
            with nogil:
                self.chan = command_channel_socket_new()
        elif self.chan_type == "TCP":
            with nogil:
                self.chan = command_channel_socket_tcp_new(0, 1)
        else:
            print("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[LOCAL | SHM | VSOCK | TCP]")
            exit(0)

    def create_worker_channel(self, listen_port):
        if self.chan_type == "LOCAL":
            self.chan = command_channel_min_worker_new(listen_port)
        elif self.chan_type == "SHM":
            self.chan = command_channel_shm_worker_new(listen_port)
        elif self.chan_type == "VSOCK":
            self.chan = command_channel_socket_worker_new(listen_port)
        elif self.chan_type == "TCP":
            self.chan = command_channel_socket_tcp_new(listen_port, 0)
        else:
            print("Unsupported AVA_CHANNEL type (export AVA_CHANNEL=[LOCAL | SHM | VSOCK | TCP]")
            exit(0)

    def buffer_size(self, size):
        return command_channel_buffer_size(self.chan, size)

    def new_command(self, command_struct_size, data_region_size):
        real_command_struct_size = <uint64_t>command_struct_size
        real_data_region_size = <uint64_t>data_region_size
        with nogil:
            ret_cmd = command_channel_new_command(self.chan, real_command_struct_size, real_data_region_size)
        new_cmd = Command_base_c()
        new_cmd.__set_ptr(self.chan, ret_cmd)
        return new_cmd

    def receive_command(self):
        with nogil:
            ret_cmd = command_channel_receive_command(self.chan)
        new_cmd = Command_base_c()
        new_cmd.__set_ptr(self.chan, ret_cmd)
        return new_cmd

    def __del__(self):
        with nogil:
            command_channel_free(self.chan)

    def __get_sizeof_tf_cmd(self):
        return sizeof(py_tf_cmd)

cdef class Vsock_c:
    cdef int listen_fd
    cdef int client_fd
    cdef int guest_cid
    cdef cmd_channel.command_base *msg
    cdef cmd_channel.command_channel *chan

    def __init__(self):
        self.listen_fd = init_manager_vsock()

    def poll_client(self):
        msg = poll_client(self.listen_fd, &self.client_fd, &self.guest_cid)
        new_cmd = Command_base_c()
        new_cmd.__set_ptr(self.chan, msg)
        return new_cmd

    def respond_client(self, worker_id):
        respond_client(self.client_fd, <int>worker_id)

    def close_client(self):
        close_client(self.client_fd)

    def close(self):
        close_client(self.listen_fd)

    def __get_guest_cid(self):
        return self.guest_cid
