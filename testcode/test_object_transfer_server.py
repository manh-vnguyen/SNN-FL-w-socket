import socket
import msgpack
import struct  # For packing the size
import time

# Model parameters
model_params = {
    "weights": [0.1, 0.2, 0.3],
    "bias": 0.05
}

# Serialize the data
serialized_params = msgpack.packb(model_params)

# Pack the size of the serialized data
packed_size = struct.pack('!I', len(serialized_params))

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# Bind the socket to the port
server_socket.bind(('localhost', 12345))
server_socket.listen(1)

print("Waiting for a connection...")
connection, client_address = server_socket.accept()

try:
    # Send the size of the serialized data
    connection.sendall(packed_size)
    
    # Send the serialized data
    connection.sendall(serialized_params)

    time.sleep(10)
finally:
    # Clean up the connection
    connection.close()
