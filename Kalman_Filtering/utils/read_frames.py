import struct
import logging as log
from google.protobuf.message import DecodeError
# from a42_proto.frame import Frame
# from a42 import a42_proto
from a42.frame import Frame
def read_length_delimited_frames(path):
    """Liest length-delimited protobuf Frames aus Datei."""
    """Yield betterproto-Frames aus einer length-delimited .pb Datei."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                return
            size = struct.unpack("<I", hdr)[0]
            blob = f.read(size)
            if len(blob) < size:
                return
            # betterproto: parse()
            yield Frame().parse(blob)
