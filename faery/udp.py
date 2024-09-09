from socket import socket, AF_INET, AF_INET6, SOCK_DGRAM

from faery.output import EventOutput
from faery.stream_types import Event, Events

import numpy as np

class UdpEventOutput(EventOutput):

    def __init__(self, server_address:str, server_port:int, include_timestamps:bool=False, **kwargs):
        self.clientSocket = socket(AF_INET, SOCK_DGRAM)
        # IPv6: AF_INET6 and addr tuple becomes (host, port, flowinfo, scope_id) the former 2 optional.
        self.clientSocket.settimeout(1)
        self.addr = (server_address, server_port)   
        self.include_ts = include_timestamps
        # Buffer = 512 bytes / 4 bytes per event. Timestamps, if sent, are other 4 bytes, and another element in the buffer.
        self.max_buffer_size = 512/4 
        self.buffer: np.ndarray = np.empty(int(self.max_buffer_size), dtype=np.uint32)
        self.buffer_idx: int = 0
        self.kwargs = kwargs

    def encode_spif(self, event: Event) -> int:
        message = int(event['y'])
        message |= (int(event['p']) << 15)
        message |= (int(event['x']) << 16)
        message |= (int(not self.include_ts) << 31)   # 0: present, 1: absent
        return message  

    def apply(self, data: Events):
        for event in data:
            message = self.encode_spif(event)
            self.buffer[self.buffer_idx] = message
            self.buffer_idx += 1
            if self.include_ts:
                timestamp = event['t']
                self.buffer[self.buffer_idx] = timestamp
                self.buffer_idx += 1
            if self.buffer_idx >= self.max_buffer_size:
                self.clientSocket.sendto(self.buffer.tobytes(), self.addr)
                self.buffer_idx = 0
        if self.buffer_idx > 0:
            self.clientSocket.sendto(self.buffer[0:self.buffer_idx].tobytes(), self.addr)
            self.buffer_idx = 0

