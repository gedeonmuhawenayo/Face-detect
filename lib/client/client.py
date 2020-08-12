import pickle
import requests
from PIL import Image
import numpy as np
import time
from scipy import misc

import os
from lib.utils import load_lib

face_cascade = "/home/dsvn/workspace/openface/models/dlib/haarcascade_frontalface_default.xml"
networkModel = "/home/dsvn/workspace/openface/models/openface/nn4.small2.v1.t7"
dlibFacePredictor = "/home/dsvn/workspace/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
dim = 96
HOST = "192.168.0.210"
PORT = 2013
output = "output_luong_test"
lib = load_lib(face_cascade=face_cascade, networkModel=networkModel, dlibFacePredictor=dlibFacePredictor)
if not os.path.exists(output):
    os.makedirs(output)

#for test for Luong
import glob
list_img = glob.glob("/home/dsvn/Documents/congnt/ID_Cards/idcards/*/*")
for img in list_img:
	print(img)
	im = np.asarray(Image.open(img))
	list_face = lib.face_detect(im)
	if list_face is None:
	    print("no face")
	    continue
	for i in range(len(list_face)):
	    misc.imsave(output+"/test_"+str(time.time())+".png", list_face[i])
	    face_align = lib.align_function(np.array(list_face[i]))
	    if face_align is not None:
	        data = pickle.dumps(face_align)
	        url = "http://"+HOST+":"+str(PORT)
	        requests.post(url=url,data=data)
'''
img = sys.argv[1]
im = np.asarray(Image.open(img))
list_face = lib.face_detect(im)
if list_face is None:
    print("no face")
    exit(1)
for i in range(len(list_face)):
    face_align = lib.align_function(np.array(list_face[i]))
    if face_align is not None:
        data = pickle.dumps(face_align)
        url = "http://"+HOST+":"+str(PORT)
        requests.post(url=url,data=data)
'''
