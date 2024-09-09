import faery
from socket import socket, AF_INET, SOCK_DGRAM

def decode_spif(b):
    event_y = b[0] + ((b[1] & 0x7F) << 8) 
    pol = bool(b[1] >> 7)
    event_x = b[2] + ((b[3] & 0x7F) << 8) 
    ts = not bool(b[3] >> 7)
    return {'x': event_x, 'y': event_y, 'p': pol, 'ts': ts}


def test_udp_event_stream():

    serverSocket = socket(AF_INET, SOCK_DGRAM)
    serverSocket.bind(("localhost", 12345))

    faery.read_file(
        "tests/data/sample.csv").output(
            faery.udp.UdpEventOutput(server_address="localhost", 
                                     server_port=12345))
    message = serverSocket.recvfrom(1024)[0]
    f = open("tests/data/sample.csv", "r")
    for i in range(0, len(message), 4):
        event_from_udp = decode_spif(message[i:i+4])
        event_from_file = f.readline().strip().split(",")
        assert event_from_udp['x'] == int(event_from_file[1])
        assert event_from_udp['y'] == int(event_from_file[2])
        assert event_from_udp['p'] == bool(int(event_from_file[3]))
        assert event_from_udp['ts'] == False
    f.close()
    serverSocket.close()

def test_udp_event_stream_with_ts():

    serverSocket = socket(AF_INET, SOCK_DGRAM)
    serverSocket.bind(("localhost", 12345))

    faery.read_file(
        "tests/data/sample.csv").output(
            faery.udp.UdpEventOutput(server_address="localhost", 
                                     server_port=12345,
                                     include_timestamps=True))
    message = serverSocket.recvfrom(1024)[0]
    f = open("tests/data/sample.csv", "r")
    for i in range(0, len(message), 8):
        event_from_udp = decode_spif(message[i:i+4])
        event_from_file = f.readline().strip().split(",")
        assert int.from_bytes(message[i+4:i+8], byteorder='little', signed=False) == int(event_from_file[0])
        assert event_from_udp['x'] == int(event_from_file[1])
        assert event_from_udp['y'] == int(event_from_file[2])
        assert event_from_udp['p'] == bool(int(event_from_file[3]))
        assert event_from_udp['ts'] == True
    f.close()
    serverSocket.close()


