import os, sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../../common/python'))
from cmd_channel import *

def init_guestlib():
    chan = Command_channel_c()
    chan.create_guestlib_channel()
    return chan
