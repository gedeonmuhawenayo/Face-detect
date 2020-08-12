import socket
import sys
import pickle
import time
from scipy import misc

import os
from lib.utils import load_lib
from lib import  cfg


face_cascade = cfg.face_cascade
networkModel = cfg.networkModel
dlibFacePredictor = cfg
dim = 96
HOST = "192.168.0.210"
PORT = 2013
output = "output"
if not os.path.exists(output):
    os.makedirs(output)

lib = load_lib(face_cascade=face_cascade, network_model=networkModel,
               dlib_face_predictor=dlibFacePredictor)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
# Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()
     
print('Socket bind complete')
 
# Start listening on socket
s.listen(10)
print('Socket now listening')
 
# now keep talking with the client
while 1:
    # wait to accept a connection - blocking call
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    data = conn.recv(4096 * 4096)
    if data is not None:
        face_align = pickle.loads(data)
        path = "test_"+str(time.time())+".png"
        print(path)
        misc.imsave(output + "/" + path, face_align)
        vector = lib.get_vector(face_align)
        lib.save_redis(path, vector)
     
s.close()
