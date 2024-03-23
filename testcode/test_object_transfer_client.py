import socket
import struct
import msgpack

def receive_full_message(sock, expected_size):
    received_data = b''
    while len(received_data) < expected_size:
        try:
            more_data = sock.recv(expected_size - len(received_data))
            if not more_data:
                raise Exception("Server closed the connection unexpectedly.")
            received_data += more_data
        except socket.timeout:
            raise Exception("Timed out waiting for data from server.")
    return received_data

# Create socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5.0)  # Set timeout for socket operations

try:
    # Connect to server
    sock.connect(('localhost', 12345))
    
    # Receive the size of the incoming data
    data_size_packed = receive_full_message(sock, 4)
    data_size = struct.unpack('!I', data_size_packed)[0] + 1

    # Receive the actual data
    serialized_data = receive_full_message(sock, data_size)
    print("Data received successfully.")

    # Deserialize the data
    model_params = msgpack.unpackb(serialized_data)
    print("Received model parameters:", model_params)

except Exception as e:
    print("Error:", e)
finally:
    sock.close()
