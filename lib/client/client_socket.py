import socket
import pickle

HOST = 'localhost'
PORT = 8888
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
arr = ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
data_string = pickle.dumps(arr)
s.send(data_string)
