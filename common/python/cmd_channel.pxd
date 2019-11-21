# cython: language_level=3
cimport cython
from libc.stdint cimport *

cdef extern from "../../include/cmd_channel.h":
    cdef struct command_channel:
        pass

    cdef struct command_base:
        uint8_t api_id;
        uint8_t vm_id;
        uintptr_t command_type;
        int64_t thread_id;
        int8_t flags;
        uintptr_t command_id;
        size_t command_size;
        void* data_region;
        size_t region_size;
        char reserved_area[64];

    cdef command_channel* command_channel_shm_new() nogil;
    cdef command_channel* command_channel_min_new() nogil;
    cdef command_channel* command_channel_socket_new() nogil;

    cdef command_channel* command_channel_shm_worker_new(int listen_port) nogil;
    cdef command_channel* command_channel_min_worker_new(int listen_port) nogil;
    cdef command_channel* command_channel_socket_worker_new(int listen_port) nogil;
    cdef command_channel* command_channel_socket_tcp_new(int worker_port, int is_guest) nogil;

    cdef void command_channel_free(command_channel* c) nogil;
    cdef size_t command_channel_buffer_size(const command_channel* chan, size_t size) nogil;
    cdef command_base* command_channel_new_command(command_channel* chan, size_t command_struct_size, size_t data_region_size) nogil;
    cdef void* command_channel_attach_buffer(command_channel* chan, command_base* cmd, const void* buffer, size_t size) nogil;
    cdef void command_channel_send_command(command_channel* chan, command_base* cmd) nogil;
    cdef void command_channel_transfer_command(command_channel* chan, const command_channel* source_chan, const command_base* cmd) nogil;
    command_base* command_channel_receive_command(command_channel* chan) nogil;
    void* command_channel_get_buffer(const command_channel* chan, const command_base* cmd, const void* buffer_id) nogil;
    void* command_channel_get_data_region(const command_channel* c, const command_base* cmd) nogil;
    void command_channel_free_command(command_channel* chan, command_base* cmd) nogil;
    void command_channel_print_command(const command_channel* chan, const command_base* cmd) nogil;

cdef struct py_tf_cmd:
    command_base base
    uintptr_t object_id
    uintptr_t buffer_id[12]
    uintptr_t buffer_len[12]

cdef extern from "../../worker/include/worker.h":
    cdef int init_manager_vsock();
    cdef command_base *poll_client(int listen_fd, int *client_fd, int *guest_cid);
    cdef void respond_client(int client_fd, int worker_id);
    cdef void close_client(int client_fd);

cdef extern from "../../include/guest_mem.h":
    cdef struct param_block_info:
        uintptr_t param_local_offset;
        uintptr_t param_block_size;
