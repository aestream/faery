from socket import socket, AF_INET, AF_INET6, SOCK_DGRAM

from faery.output import EventOutput
from faery.stream_types import Event, Events

class UdpEventOutput(EventOutput):

    def __init__(self, server_address:str, server_port:int, include_timestamps:bool=False, **kwargs):
        self.clientSocket = socket(AF_INET, SOCK_DGRAM)
        # IPv6: AF_INET6 and addr tuple becomes (host, port, flowinfo, scope_id) the former 2 optional.
        self.clientSocket.settimeout(1)
        self.addr = (server_address, server_port)   
        self.include_ts = include_timestamps
        # Buffer = 512 bytes / 4 bytes per event. Timestamps, if sent, are other 4 bytes, and another element in the buffer.
        self.max_buffer_size = 512/4 
        self.buffer: list[bytes] = []
        self.kwargs = kwargs

    def encode_spif(self, event: dict) -> bytes:
        # Check if the coordinates can be sent in the 32 bit packet. If not, set them to the maximum value.
        # Should we raise an exception instead? 1920x1080 is the maximum resolution for DVS for now.
        if (event['y'] > 2**15):
            event['y'] = 2**15 - 1
        if (event['x'] > 2**15):
            event['x'] = 2**15 - 1
        message = 0
        message = message + int(event['y'])
        message = message + (int(event['p']) << 15)
        message = message + (int(event['x']) << 16)
        message = message + (int(not self.include_ts) << 31)   # 0: present, 1: absent
        message = message.to_bytes(
            length=4, 
            byteorder='little', 
            signed=False
        )
        return message

    def apply(self, data: Events):
        for event in data:
            message = self.encode_spif(event)
            self.buffer.append(message)
            if self.include_ts:
                timestamp_b = int(event['t']).to_bytes(length=4, byteorder='little', signed=False) 
                self.buffer.append(timestamp_b)
            if len(self.buffer) >= self.max_buffer_size:
                self.clientSocket.sendto(b''.join(self.buffer), self.addr)
                self.buffer.clear()
        #If the buffer is not empty, send the remaining data
        if len(self.buffer) > 0:
            self.clientSocket.sendto(b''.join(self.buffer), self.addr)
            self.buffer.clear() 

