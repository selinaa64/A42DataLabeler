import struct
import logging
from google.protobuf.message import DecodeError
from a42.frame_pb2 import Frame

def read_length_delimited_frames(path):
    """Liest length-delimited protobuf Frames aus Datei."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                return
            size = struct.unpack("<I", hdr)[0]
            blob = f.read(size)
            if len(blob) < size:
                log.warning(f"{path}: erwartete {size} Bytes, nur {len(blob)} gelesen")
                return
            frame = Frame()
            try:
                frame.ParseFromString(blob)
            except DecodeError as e:
                log.error(f"{path}: Parse-Fehler: {e}")
                return
            yield frame