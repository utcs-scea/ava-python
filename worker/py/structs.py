# must sync with attribute.h and ctype_util.h

from ctypes import *

# typedef

BOOLEAN = c_char

# _MINI_TASK_NODE

class MINI_TASK_NODE(Structure):
    _fields_ = [("vm_id",          c_int),
                ("rt_type",        c_uint, 8),
                ("data_ptr",       c_ulonglong),

                ("node_id",        c_longlong),
                ("IsSwap",         BOOLEAN),
                ("IsHighPriority", BOOLEAN)
               ]

STOP_HANDLER = -100  # assign to vm_id

class PARAM_BLOCK_INFO(Structure):
    _fields_ = [
                   ("param_local_offset", c_uint64),
                   ("param_block_size", c_uint64)
               ]

class RESERVED_AREA(Union):
    _fields_ = [
                   ("string", c_char * 64),
                   ("uint64", c_uint64 * 8),
                   ("uint8",  c_uint8 * 64),
                   ("pb_info", PARAM_BLOCK_INFO)
               ]
class COMMAND_BASE(Structure):
    _fields_ = [
                   ("api_id", c_uint8),
                   ("vm_id",  c_uint8),
                   ("command_type", c_uint64),
                   ("thread_id", c_int64),
                   ("flags", c_int, 8),
                   ("command_id", c_uint64),
                   ("command_size", c_size_t),
                   ("data_region", c_void_p),
                   ("region_size", c_size_t),
                   ("reserved_area", RESERVED_AREA)
               ]


